from __future__ import annotations

import json
import asyncio
import logging
import warnings
import hashlib
import base64
import random
from functools import wraps
from io import BytesIO
from functools import wraps
from traceback import format_exception
from typing import Any, Callable, Coroutine, Union, Generator
from datetime import datetime, UTC
import numpy as np
from PIL import Image

from starlette.concurrency import run_in_threadpool

from bson import ObjectId
from bson.errors import InvalidId

from api.models import StreamResponse

NoArgsNoReturnFuncT = Callable[[], None]
NoArgsNoReturnAsyncFuncT = Callable[[], Coroutine[Any, Any, None]]
ExcArgNoReturnFuncT = Callable[[Exception], None]
ExcArgNoReturnAsyncFuncT = Callable[[Exception], Coroutine[Any, Any, None]]
NoArgsNoReturnAnyFuncT = Union[NoArgsNoReturnFuncT, NoArgsNoReturnAsyncFuncT]
ExcArgNoReturnAnyFuncT = Union[ExcArgNoReturnFuncT, ExcArgNoReturnAsyncFuncT]
NoArgsNoReturnDecorator = Callable[[NoArgsNoReturnAnyFuncT], NoArgsNoReturnAsyncFuncT]


async def _handle_func(func: NoArgsNoReturnAnyFuncT) -> None:
    if asyncio.iscoroutinefunction(func):
        await func()
    else:
        await run_in_threadpool(func)


async def _handle_exc(exc: Exception, on_exception: ExcArgNoReturnAnyFuncT | None) -> None:
    if on_exception:
        if asyncio.iscoroutinefunction(on_exception):
            await on_exception(exc)
        else:
            await run_in_threadpool(on_exception, exc)


def repeat_every(
    *,
    seconds: float,
    wait_first: float | None = None,
    logger: logging.Logger | None = None,
    raise_exceptions: bool = False,
    max_repetitions: int | None = None,
    on_complete: NoArgsNoReturnAnyFuncT | None = None,
    on_exception: ExcArgNoReturnAnyFuncT | None = None,
) -> NoArgsNoReturnDecorator:
    """
    reference: https://github.com/dmontagu/fastapi-utils/blob/master/fastapi_utils/tasks.py
    This function returns a decorator that modifies a function so it is periodically re-executed after its first call.

    The function it decorates should accept no arguments and return nothing. If necessary, this can be accomplished
    by using `functools.partial` or otherwise wrapping the target function prior to decoration.

    Parameters
    ----------
    seconds: float
        The number of seconds to wait between repeated calls
    wait_first: float (default None)
        If not None, the function will wait for the given duration before the first call
    logger: Optional[logging.Logger] (default None)
        Warning: This parameter is deprecated and will be removed in the 1.0 release.
        The logger to use to log any exceptions raised by calls to the decorated function.
        If not provided, exceptions will not be logged by this function (though they may be handled by the event loop).
    raise_exceptions: bool (default False)
        Warning: This parameter is deprecated and will be removed in the 1.0 release.
        If True, errors raised by the decorated function will be raised to the event loop's exception handler.
        Note that if an error is raised, the repeated execution will stop.
        Otherwise, exceptions are just logged and the execution continues to repeat.
        See https://docs.python.org/3/library/asyncio-eventloop.html#asyncio.loop.set_exception_handler for more info.
    max_repetitions: Optional[int] (default None)
        The maximum number of times to call the repeated function. If `None`, the function is repeated forever.
    on_complete: Optional[Callable[[], None]] (default None)
        A function to call after the final repetition of the decorated function.
    on_exception: Optional[Callable[[Exception], None]] (default None)
        A function to call when an exception is raised by the decorated function.
    """

    def decorator(func: NoArgsNoReturnAnyFuncT) -> NoArgsNoReturnAsyncFuncT:
        """
        Converts the decorated function into a repeated, periodically-called version of itself.
        """

        @wraps(func)
        async def wrapped() -> None:
            async def loop() -> None:
                if wait_first is not None:
                    await asyncio.sleep(wait_first)

                repetitions = 0
                while max_repetitions is None or repetitions < max_repetitions:
                    try:
                        if logger is not None:
                            logger.info(f"{"Running":<10}: {func.__name__} cron job {datetime.now(UTC)}")
                        await _handle_func(func)
                        if logger is not None:
                            logger.info(f"{"Finish":<10}: {func.__name__} {"job":<8} {datetime.now(UTC)}")


                    except Exception as exc:
                        if logger is not None:
                            warnings.warn(
                                "'logger' is to be deprecated in favor of 'on_exception' in the 1.0 release.",
                                DeprecationWarning,
                            )
                            formatted_exception = "".join(format_exception(type(exc), exc, exc.__traceback__))
                            logger.error(formatted_exception)
                        if raise_exceptions:
                            warnings.warn(
                                "'raise_exceptions' is to be deprecated in favor of 'on_exception' in the 1.0 release.",
                                DeprecationWarning,
                            )
                            raise exc
                        await _handle_exc(exc, on_exception)

                    repetitions += 1
                    await asyncio.sleep(seconds)

                if on_complete:
                    await _handle_func(on_complete)

            asyncio.ensure_future(loop())

        return wrapped

    return decorator


def generate_hash(input_string: str) -> str:
    """Generate an MD5 hash for a given input string.

    Args:
        input_string (str): The string to hash.

    Returns:
        str: The resulting MD5 hash in hexadecimal format.
    """
    return hashlib.md5(input_string.encode()).hexdigest()

def generate(txt: str, primary: int | None = None, secondary: int | None = None) -> str:
        """Generate a 5x5 identicon image based on the provided text.

        Args:
            txt (str): The text to be hashed to generate the identicon.
            primary (int, optional): The primary color for the identicon in RGB integer format. Defaults to 0XB4F8C8.
            secondary (int, optional): The secondary color for the identicon in RGB integer format. Defaults to 0xFFFFFF.

        Returns:
            np.ndarray: A 5x5 identicon image represented as a NumPy array.

        Raises:
            ValueError: If primary or secondary colors are not in the expected format.
        """
        if primary is None:
            primary = 0xFFFFFF
        if secondary is None:
            secondary = random.choice([0x00FF00, 0x800080, 0xFF0000, 0xFFC0CB, 0x0000FF])
        # Validate color format and convert to RGB tuple if necessary
        def validate_and_convert_color(color):
            if isinstance(color, tuple):
                if len(color) != 3:
                    raise ValueError("Color tuples must have three components.")
                return color
            elif isinstance(color, int):
                if not (0 <= color <= 0xFFFFFF):
                    raise ValueError("Color integers must be between 0x0 and 0xFFFFFF.")
                return ((color >> 16) & 255, (color >> 8) & 255, color & 255)
            else:
                raise TypeError("Color must be a tuple or an integer.")

        primary_color = validate_and_convert_color(primary)
        secondary_color = validate_and_convert_color(secondary)
        grid_size = 5
        identicon = np.zeros((grid_size, grid_size, 3), dtype=np.uint8)

        hash_code = generate_hash(txt)

        for i in range(grid_size+1 // 2):
            for j in range(grid_size):
                # Converting hash character to a number
                hash_value = int(hash_code[i * grid_size + j], 16)
                # If the value is even, we color the pixel
                if hash_value % 2 == 0:
                    # Use the next three characters to determine the color
                    colour = primary_color
                    identicon[j, i] = colour
                    identicon[j, grid_size - i - 1] = colour  # Mirror the color
                else:
                    colour = secondary_color
                    identicon[j, i] = colour
                    identicon[j, grid_size - i - 1] = colour  # Mirror the color


        i, j = identicon.shape[:2]
        h, w = 500 // i, 500 // j
        identicon = np.repeat(identicon, h, axis=0)
        identicon = np.repeat(identicon, w, axis=1)

        # Image to PIL
        img = Image.fromarray(identicon)

        # convert image buffer to webp
        buffered = BytesIO()
        img.save(buffered, format="WEBP")
        img_bytes = buffered.getvalue()
        
        return f"data:image/webp;base64,{base64.b64encode(img_bytes).decode("utf-8")}"


def to_object_id(id : str | ObjectId) -> ObjectId:
    try: 
        if isinstance(id, str):
            return ObjectId(id)
        return id
    except InvalidId as e:
        raise e


logger = logging.getLogger('uvicorn.error')

def handleStreamResponse(
    *,
    retry: int = 1000,
    include_end: bool = False
):
    """
    Decorator for handling streaming responses in FastAPI endpoints.
    
    Args:
        f: The async generator function to wrap
        retry: Milliseconds to wait before retry on connection failure
        include_end: Whether to include an end event in the stream
        
    Returns:
        Wrapped async generator function that yields properly formatted SSE events
    """
    def decorator(func: Callable[..., Generator[Any, None]]):
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Generator[dict, None]:
            try:
                for item in func(*args, **kwargs):
                    if hasattr(item, 'streamingResponse'):
                        yield item.streamingResponse()
                    elif isinstance(item, dict):
                        yield f"{json.dumps(item)}"
                    elif isinstance(item, str):
                        yield item
                if include_end:
                    yield StreamResponse(
                        type="end",
                        data={}
                    )
                    
            except Exception as e:
                logger.error(f"Streaming Event Error: {e}")
                yield {
                    "event": "error",
                    "id": "error_message",
                    "retry": retry,
                    "data": str(e)
                }
        return wrapper
    return decorator
    
