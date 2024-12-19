import { useToast } from '@/hooks/use-toast';
import { CheckIcon, X , Ban} from "lucide-react";
import { ToastClose } from '@radix-ui/react-toast';



interface ToastMessage {
    message: string;
  }
  
const WarningMessage = {
    Description: ({ message }: ToastMessage) => (
      <div className="space-x-3 flex items-center text-yellow-500">
        <Ban className="h-5 w-5" />
        <span>{message}</span>
      </div>
    ),
    Action: () => (
      <ToastClose/>
    )
  };

const ErrorMessage = {
  Description: ({ message }: ToastMessage) => (
    <div className="space-x-3 flex items-center text-red-500">
      <X className="h-5 w-5" />
      <span>{message}</span>
    </div>
  ),
  Action: () => (
    <ToastClose/>
  )
};


const SuccessfulMessage = {
  Description: ({ message }: ToastMessage) => (
    <div className="space-x-3 flex items-center text-emerald-500">
      <CheckIcon className="h-5 w-5" />
      <span>{message}</span>
    </div>
  ),
  Action: () => (
    <ToastClose/>
  )
};


export const useNotification = () => {
  const { toast } = useToast();

  return {
    showSuccess: (success : string) => {
      toast({
        description: <SuccessfulMessage.Description message={success} />,
        action: <SuccessfulMessage.Action />,

      });
    },
    
    showWarning: (warning: string) => {
      toast({
        description: <WarningMessage.Description message={warning} />,
        action: <WarningMessage.Action />,
      });
    },
    
    showError: (err: string) => {
      toast({
        description: <ErrorMessage.Description message={err} />,
        action: <ErrorMessage.Action />,
      });
    },
  };
};
