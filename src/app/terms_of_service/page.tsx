'use server';
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area";
import { Separator } from "@/components/ui/separator";
import Link from "next/link";
import { ChevronLeft } from "lucide-react";

export default async function TermOfService() {
    return (<>

        <div className="container mx-auto py-8 px-4 max-w-4xl">
        <div className="mb-6">
        <Link href="/">
          <Button variant="ghost" className="flex items-center gap-2">
            <ChevronLeft className="h-4 w-4" />
            Back to Home
          </Button>
        </Link>
      </div>
        <h1 className="text-3xl font-bold text-center mb-8">Terms of Service</h1>
        
        <Alert className="mb-6 ">
          <AlertDescription>
            Before using the service, users must agree to the following terms:
            This service is offered as a research preview and includes only limited safety measures. 
            It may produce offensive content. Users are prohibited from uploading any private or sensitive information. 
            The service collects dialogue data, and reserves the right to distribute this data under a Creative Commons Attribution (CC-BY) license.
          </AlertDescription>
        </Alert>
  
        <div className="text-sm text-muted-foreground mb-6">
          The content on the website does not reflect the perspectives of the creators or affiliated institutions engaged.
        </div>
  
        <ScrollArea className="h-[600px] pr-4">
          <div className="space-y-6">
            <section>
              <h2 className="text-xl font-semibold mb-3">1. Acceptance of Terms</h2>
              <p className="text-sm leading-relaxed">
                By accessing or using the RedTeam Arena service, you agree to be bound by these Terms of Service. 
                If you disagree with any part of the terms, you may not access the service.
              </p>
            </section>
  
            <Separator />
  
            <section>
              <h2 className="text-xl font-semibold mb-3">2. Description of Service</h2>
              <p className="text-sm leading-relaxed">
                RedTeam Arena is a research preview service that allows users to interact with an AI model. 
                The service is provided "as is" and may be subject to changes or termination at any time without notice.
              </p>
            </section>
  
            <Separator />
  
            <section>
              <h2 className="text-xl font-semibold mb-3">3. User Responsibilities</h2>
              <ul className="list-disc pl-6 text-sm leading-relaxed space-y-2">
                <li>You must be at least 18 years old to use this service.</li>
                <li>You are responsible for maintaining the confidentiality of your account and password.</li>
                <li>You agree not to use the service for any illegal or unauthorized purpose.</li>
                <li>You must not upload any private or sensitive information to the service.</li>
              </ul>
            </section>
  
            <Separator />
  
            <section>
              <h2 className="text-xl font-semibold mb-3">4. Content and Conduct</h2>
              <p className="text-sm leading-relaxed">
                The service may produce offensive or inappropriate content. Users are advised to use discretion 
                and judgment when interacting with the AI model. RedTeam Arena is not responsible for any 
                content generated by the AI or submitted by users.
              </p>
            </section>
  
            <Separator />
  
            <section>
              <h2 className="text-xl font-semibold mb-3">5. Data Collection and Use</h2>
              <p className="text-sm leading-relaxed">
                The service collects dialogue data from user interactions. By using the service, you grant 
                RedTeam Arena the right to collect, store, and potentially distribute this data under a 
                Creative Commons Attribution (CC-BY) license.
              </p>
            </section>
  
            <Separator />
  
            <section>
              <h2 className="text-xl font-semibold mb-3">6. Limitation of Liability</h2>
              <p className="text-sm leading-relaxed">
                RedTeam Arena shall not be liable for any indirect, incidental, special, consequential or 
                punitive damages, or any loss of profits or revenues, whether incurred directly or indirectly, 
                or any loss of data, use, goodwill, or other intangible losses, resulting from your access 
                to or use of or inability to access or use the service.
              </p>
            </section>
  
            <Separator />
  
            <section>
              <h2 className="text-xl font-semibold mb-3">7. Changes to Terms</h2>
              <p className="text-sm leading-relaxed">
                We reserve the right to modify or replace these terms at any time. It is your responsibility 
                to check the Terms periodically for changes.
              </p>
            </section>
          </div>
        </ScrollArea>
      </div>
      </>)
}