import time
import aiohttp
import asyncio

from urllib import parse
from typing import Dict, Any
from datetime import datetime, UTC

from fastapi import APIRouter, HTTPException, status, Request

from api.core.config import settings
from api.backend_ping_test import db_ping_server

async def check_model_service(url : str, model_endpoint: str) -> Dict[str, str]:
    """ helper function that pings public services """
    try:
        
        endpoint = parse.urljoin(url, model_endpoint)
        async with aiohttp.ClientSession() as session:
            start_time = time.time()
            async with session.get(endpoint, timeout=5) as response:
                end_time = time.time()
                return {
                    "endpoint" : endpoint,
                    "status": response.status,
                    "latency": f"{(end_time - start_time):.2f}s"
                }
    except Exception as e:
        return {"endpoint" : endpoint, "status": "error", "message": str(e)}

router = APIRouter()

@router.get("/health/")
async def health_check(request : Request) -> Dict[str, Any]:
    """ check the health of mongodb latency and ping public routes """
    start_time = datetime.now(UTC)
    base_url = str(request.base_url)
    # Run all health checks concurrently
    ping = await db_ping_server()
    
    checks = await asyncio.gather(
        check_model_service(base_url, f"game/?s=0"),
        check_model_service(base_url, f"model"),
        return_exceptions=False
    )
    
    end_time = datetime.now(UTC)
    response_time = (end_time - start_time).total_seconds()
    
    health_status = {
        "status": "healthy",  # Will be updated based on checks
        "timestamp": datetime.now(UTC).isoformat(),
        "response_time": f"{response_time:.2f}s",
        "services": {
            "mongoDB": ping,
            "endpoints" : [{ "service" : check } for check in checks],
            "api": {
                "version": "0.1.0", 
            }
        },
        "environment": settings.ENVIRONMENT  # Add this during app initialization
    }
    
    # Determine overall status
    if any(check.get("status") == "error" for check in [
        checks[0],  # MongoDB
        checks[1]   # Model Service
    ]):
        health_status["status"] = "error"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=health_status
        )

    
    return health_status