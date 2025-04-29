'use client';

import React, { useState, useEffect, useCallback, useRef } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription } from '@/components/ui/card';
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert';
import { ActivityLog } from '@/components/activity-log'; // Custom icon component
import { useToast } from "@/hooks/use-toast";
import { sendEmail } from '@/services/email';
import type { Activity } from '@/types/activity';
import { AlertCircle, ShieldCheck, Cpu, Network, Server, Database, Terminal } from 'lucide-react';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Skeleton } from '@/components/ui/skeleton';

// Simulate data generation (replace with actual data source)
const generateFakeData = (): Activity => {
  const isSuspicious = Math.random() < 0.2; // 20% chance of suspicious activity
  const timestamp = new Date();
  const activities = [
    'User Login',
    'File Access',
    'Network Connection Attempt',
    'Database Query',
    'Configuration Change',
    'API Call',
  ];
  const systems = ['Auth Server', 'File Server', 'Web App', 'Database Cluster', 'Firewall', 'API Gateway'];
  const descriptions = [
    'Multiple failed login attempts from IP 192.168.1.100',
    'Accessed sensitive financial_report.xlsx',
    'Attempted connection to known malicious IP 10.0.0.5',
    'Executed DROP TABLE command on users table',
    'Disabled firewall rule #3',
    'Anomalous high volume of requests to /admin endpoint',
    'Successful login from new device and location',
    'Downloaded large file data_backup.zip',
    'Established outbound connection to port 8080',
    'Performed SELECT * query on customer_pii table',
    'Enabled debug mode on production server',
    'Received large payload in API request',
    'User activity detected outside normal business hours.',
    'Configuration backup process initiated.',
    'Standard API health check received.',
    'User password reset successfully completed.',
    'Read access to non-sensitive configuration file.',
  ];

  const descriptionIndex = Math.floor(Math.random() * descriptions.length);
  const systemIndex = Math.floor(Math.random() * systems.length);
  const activityIndex = Math.floor(Math.random() * activities.length);

  // Make some descriptions more likely to be suspicious
  let generatedType: 'normal' | 'suspicious' = isSuspicious ? 'suspicious' : 'normal';
  let generatedDescription = descriptions[descriptionIndex];

  // Simplified logic: Rely purely on the random chance + specific keywords for suspicious type
  if (generatedDescription.includes('DROP TABLE') || generatedDescription.includes('malicious IP') || generatedDescription.includes('Disabled firewall')) {
    generatedType = 'suspicious';
  } else if (generatedType === 'normal' && Math.random() < 0.05) {
     // Occasionally mark non-obvious things as suspicious if initial check was normal
     generatedType = 'suspicious';
     generatedDescription = 'Unusual pattern detected: ' + generatedDescription;
  }


  return {
    id: crypto.randomUUID(),
    timestamp: timestamp.toISOString(),
    type: generatedType,
    description: generatedDescription,
    sourceIp: `192.168.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}`,
    userId: generatedType === 'suspicious' && Math.random() > 0.5 ? 'unknown' : `user_${Math.floor(Math.random() * 1000)}`,
    system: systems[systemIndex],
    activity: activities[activityIndex],
    data: { detail: `Detail about ${activities[activityIndex]} on ${systems[systemIndex]}` }
  };
};

const getIconForSystem = (system: string) => {
  switch (system) {
    case 'Auth Server': return <Server className="h-5 w-5 text-primary" />;
    case 'File Server': return <Database className="h-5 w-5 text-green-600" />; // Keep specific color if desired, but theme preferred
    case 'Web App': return <Cpu className="h-5 w-5 text-blue-600" />; // Keep specific color if desired, but theme preferred
    case 'Database Cluster': return <Database className="h-5 w-5 text-purple-600" />; // Keep specific color if desired, but theme preferred
    case 'Firewall': return <ShieldCheck className="h-5 w-5 text-destructive" />; // Use destructive theme color
    case 'API Gateway': return <Network className="h-5 w-5 text-yellow-600" />; // Keep specific color if desired, but theme preferred
    default: return <Terminal className="h-5 w-5 text-muted-foreground" />;
  }
};


export default function Home() {
  const [activityLog, setActivityLog] = useState<Activity[]>([]);
  const [latestAlert, setLatestAlert] = useState<Activity | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const { toast } = useToast();
  const timeoutRef = useRef<NodeJS.Timeout | null>(null); // Ref to hold the timeout ID
  const isMounted = useRef(true);


  // This function handles processing a single data item
  const processSingleData = useCallback(async (data: Activity) => {
      if (!isMounted.current) return; // Don't process if unmounted

      setIsProcessing(true);
      console.log(`Processing data: ${data.id} - Type: ${data.type}`);

      await new Promise(resolve => setTimeout(resolve, 50));

      if (!isMounted.current) {
        setIsProcessing(false);
        return;
      }

      // Update activity log state immutably
      setActivityLog(prevLog => [data, ...prevLog.slice(0, 99)]);

      if (data.type === 'suspicious') {
          console.log(`Suspicious activity detected: ${data.description}`);
          setLatestAlert(data);

          try {
              // Send email alert
              await sendEmail({
                  to: 'admin@example.com',
                  subject: `Suspicious Activity Detected: ${data.activity}`,
                  html: `<p>Suspicious activity detected:</p>
                         <pre>${JSON.stringify(data, null, 2)}</pre>
                         <p>Please investigate immediately.</p>`,
              });
              toast({
                  title: "Alert Sent",
                  description: `Email notification sent for suspicious activity: ${data.description.substring(0, 50)}...`,
                  variant: "default",
              });
          } catch (error) {
              console.error("Error sending alert email:", error);
              toast({
                  title: "Alert Error",
                  description: `Failed to send email alert for ${data.id}. Check console.`,
                  variant: "destructive",
              });
          }
      }
      setIsProcessing(false);
  }, [toast]);

  // Function to schedule the next data generation
  const scheduleNextData = useCallback(() => {
      if (!isMounted.current) return; // Don't schedule if unmounted

      // Clear any existing timeout before setting a new one
      if (timeoutRef.current) {
          clearTimeout(timeoutRef.current);
      }

      const randomDelay = Math.random() * 7000; // 0ms to 7000ms (7s)
      console.log(`Next data generation in ${Math.round(randomDelay / 1000)} seconds`);

      timeoutRef.current = setTimeout(() => {
        if (!isMounted.current) return;
          const newData = generateFakeData();
          processSingleData(newData).then(() => {
              scheduleNextData();
          }).catch(error => {
             console.error("Error processing data:", error);
             scheduleNextData();
          });
      }, randomDelay);
  }, [processSingleData]);

  // Effect for initialization and cleanup
  useEffect(() => {
      isMounted.current = true;

      // Load initial data point immediately
      const initialData = generateFakeData();
      processSingleData(initialData).then(() => {
          if (isMounted.current) {
             scheduleNextData();
          }
      });

      return () => {
          isMounted.current = false;
          if (timeoutRef.current) {
              clearTimeout(timeoutRef.current);
          }
          console.log("Component unmounted, clearing data generation timeout.");
      };
  }, [processSingleData, scheduleNextData]);


  return (
    <div className="min-h-screen bg-background text-foreground p-4 md:p-8">
      <header className="mb-8 flex justify-between w-full">
        <h1 className="text-3xl font-bold text-primary flex items-center gap-2">
            <svg width="131" height="63" viewBox="0 0 131 63" fill="none" xmlns="http://www.w3.org/2000/svg" className="h-8 w-auto">
              <path fillRule="evenodd" clipRule="evenodd" d="M53.8476 1.30809C37.0207 4.38157 17.9635 14.7968 4.71513 28.16L0 32.9161L8.26916 33.275C12.8172 33.4729 21.834 33.6668 28.3068 33.7065L40.0758 33.7789L40.4447 36.2628C41.2905 41.9513 46.0198 49.1153 51.3127 52.7254C65.7997 62.6074 85.6901 54.9027 90.4053 37.5824L91.3274 34.1954L104.678 34.3116C112.02 34.376 118.159 34.5613 118.32 34.7229C119.216 35.626 104.647 45.5724 96.9487 49.3137C84.6391 55.2962 74.5581 57.6126 60.7355 57.6357C50.6392 57.6523 45.6532 56.9034 37.3538 54.1244C32.7972 52.5989 23.2563 48.1 19.2472 45.5874C14.8052 42.8032 16.7505 44.5853 22.0718 48.1759C32.5006 55.2122 43.8254 59.9032 54.7487 61.7106C63.4256 63.1464 74.4948 62.5913 82.6916 60.3082C99.369 55.6637 113.408 47.4229 126.236 34.7505L131 30.0434L128.802 29.6856C127.593 29.4888 117.463 29.1719 106.29 28.9815L85.9759 28.6352L85.7055 32.8885C85.3514 38.4521 83.3936 42.5323 79.2425 46.3588C75.2128 50.0742 71.2157 51.6112 65.583 51.6112C59.9959 51.6112 55.948 50.0696 52.0147 46.4433C49.0019 43.6661 46.4116 39.2776 45.6429 35.6485L45.2471 33.7789H51.3081H57.3697L72.8821 27.3559C81.4136 23.8228 88.5113 20.8356 88.6545 20.7177C89.2288 20.2437 86.5125 16.1164 83.7306 13.235C79.8778 9.24517 73.9673 6.36038 68.1339 5.6235C64.4937 5.16331 65.1615 5.11327 72.7116 5.27778C80.3135 5.44403 82.4692 5.71324 88.1093 7.2008C95.6308 9.18363 102.479 11.9718 108.913 15.67C111.377 17.0868 113.475 18.1642 113.574 18.0641C113.938 17.6971 103.079 10.8225 98.4606 8.49622C92.2399 5.36234 83.4546 2.46143 76.7458 1.32535C70.2856 0.231824 59.7837 0.223771 53.8476 1.30809ZM54.1772 6.69574C40.471 9.65935 28.321 15.5066 15.968 25.0854L12.5463 27.739L18.8195 28.0973C22.2697 28.2941 32.1499 28.4701 40.7755 28.4879L56.4584 28.5207L68.4344 23.4995C76.0232 20.3173 80.4104 18.1803 80.4104 17.6649C80.4104 16.4287 74.8815 12.7984 71.2374 11.6422C66.9921 10.2956 61.3218 10.7264 56.9945 12.7248C52.9534 14.5908 48.6871 19.0271 47.011 23.1055L45.6982 26.3009H43.1656H40.6329L40.9985 24.4314C42.4915 16.794 49.9292 9.24231 58.7196 6.43976C61.9902 5.39686 59.5476 5.53491 54.1772 6.69574Z" fill="currentColor"/>
            </svg>
          DarkScout AI
        </h1>
        <p className="text-black">Real-time Threat Detection PoC</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <Card className="lg:col-span-2 bg-card shadow-lg rounded-lg border border-border">
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-card-foreground">
              <ActivityLog size={20} /> Activity Log
            </CardTitle>
             <CardDescription className="text-muted-foreground">Stream of system activities (latest first).</CardDescription>
          </CardHeader>
          <CardContent>
             <ScrollArea className="h-[600px] p-4 border border-input rounded-md bg-secondary/30">
                 {activityLog.length === 0 && !isProcessing && <p className="text-muted-foreground italic text-center py-4">No activity logged yet...</p>}
                 {isProcessing && activityLog.length === 0 && (
                    <div className="space-y-3">
                        <Skeleton className="h-16 w-full" />
                        <Skeleton className="h-16 w-full" />
                        <Skeleton className="h-16 w-full" />
                    </div>
                 )}
                 {activityLog.map((activity) => (
                    <div key={activity.id} className={`mb-3 p-3 rounded-md border ${activity.type === 'suspicious' ? 'border-destructive/40 bg-destructive/10 shadow-sm' : 'border-border bg-card/50'}`}>
                      <div className="flex justify-between items-start mb-1">
                         <div className="flex items-center gap-2 flex-grow mr-2">
                            {getIconForSystem(activity.system)}
                           <span className={`font-semibold ${activity.type === 'suspicious' ? 'text-destructive' : 'text-foreground'}`}>
                             {activity.activity} on {activity.system}
                           </span>
                         </div>
                         <span className="text-xs text-muted-foreground flex-shrink-0">
                           {new Date(activity.timestamp).toLocaleString()}
                         </span>
                      </div>
                       <p className="text-sm text-foreground/90 break-words hyphens-auto" title={activity.description}>{activity.description}</p>
                       <div className="text-xs mt-1 text-muted-foreground flex flex-wrap gap-x-4 gap-y-1">
                           <span>IP: {activity.sourceIp || 'N/A'}</span>
                           <span>User: {activity.userId || 'N/A'}</span>
                       </div>
                    </div>
                 ))}
             </ScrollArea>
          </CardContent>
        </Card>

        <div className="space-y-6">
          <Card className="bg-card shadow-lg rounded-lg border border-border">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <AlertCircle className="h-6 w-6" /> Latest Alert
              </CardTitle>
               <CardDescription className="text-muted-foreground">Details of the most recent suspicious activity detected.</CardDescription>
            </CardHeader>
            <CardContent>
              {latestAlert ? (
                <Alert variant="destructive" className="shadow-inner">
                  <AlertCircle className="h-4 w-4" />
                  <AlertTitle className="font-bold">Suspicious Activity Detected!</AlertTitle>
                  <AlertDescription className="mt-2 space-y-1 text-sm">
                    <p><strong>Time:</strong> {new Date(latestAlert.timestamp).toLocaleString()}</p>
                    <p><strong>Desc:</strong> {latestAlert.description}</p>
                    <p><strong>System:</strong> {latestAlert.system}</p>
                    <p><strong>Source IP:</strong> {latestAlert.sourceIp}</p>
                    <p><strong>User:</strong> {latestAlert.userId}</p>
                  </AlertDescription>
                </Alert>
              ) : (
                <p className="text-muted-foreground italic text-center py-4">No suspicious activity detected recently.</p>
              )}
            </CardContent>
          </Card>
        </div>
      </div>
    </div>
  );
}
