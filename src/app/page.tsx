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
  // Make suspicious activity less frequent to simulate more realistic threat scenarios
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
  // Use theme colors via Tailwind classes
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
  const [isProcessing, setIsProcessing] = useState(false); // Tracks if *any* processing is happening
  const { toast } = useToast();
  const timeoutRef = useRef<NodeJS.Timeout | null>(null); // Ref to hold the timeout ID
  const isMounted = useRef(true); // Track mount status


  // This function handles processing a single data item
  const processSingleData = useCallback(async (data: Activity) => {
      if (!isMounted.current) return; // Don't process if unmounted

      setIsProcessing(true); // Indicate start of processing a single item
      console.log(`Processing data: ${data.id} - Type: ${data.type}`);

      // Simulate short processing delay
      await new Promise(resolve => setTimeout(resolve, 50));

      if (!isMounted.current) { // Check again after delay
        setIsProcessing(false);
        return;
      }

      // Update activity log state immutably
      setActivityLog(prevLog => [data, ...prevLog.slice(0, 99)]); // Keep log size manageable

      if (data.type === 'suspicious') {
          console.log(`Suspicious activity detected: ${data.description}`);
          setLatestAlert(data);

          try {
              // Send email alert
              await sendEmail({
                  to: 'admin@example.com', // Replace with actual admin email
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
      setIsProcessing(false); // Indicate end of processing
  }, [toast]); // Removed isProcessing dependency, added isMounted

  // Function to schedule the next data generation
  const scheduleNextData = useCallback(() => {
      if (!isMounted.current) return; // Don't schedule if unmounted

      // Clear any existing timeout before setting a new one
      if (timeoutRef.current) {
          clearTimeout(timeoutRef.current);
      }

      // Calculate random delay (e.g., between 3 and 20 seconds)
      // This makes the timing more varied and less predictable.
      const randomDelay = Math.random() * 17000 + 3000; // 3000ms (3s) to 20000ms (20s)
      console.log(`Next data generation in ${Math.round(randomDelay / 1000)} seconds`);

      timeoutRef.current = setTimeout(() => {
        if (!isMounted.current) return; // Check mount status before generating/processing
          const newData = generateFakeData();
          processSingleData(newData).then(() => {
              // Schedule the next one *after* the current one has been processed
              scheduleNextData();
          }).catch(error => {
             console.error("Error processing data:", error);
             // Optionally schedule next one even on error, or implement retry logic
             scheduleNextData();
          });
      }, randomDelay);
  }, [processSingleData]); // Depend on processSingleData

  // Effect for initialization and cleanup
  useEffect(() => {
      isMounted.current = true; // Component is mounted

      // Load initial data point immediately
      const initialData = generateFakeData();
      processSingleData(initialData).then(() => {
          // Then start the random scheduling if still mounted
          if (isMounted.current) {
             scheduleNextData();
          }
      });

      // Cleanup function to clear the timeout and mark as unmounted
      return () => {
          isMounted.current = false; // Mark as unmounted
          if (timeoutRef.current) {
              clearTimeout(timeoutRef.current);
          }
          console.log("Component unmounted, clearing data generation timeout.");
      };
      // Ensure dependencies are correct and stable
  }, [processSingleData, scheduleNextData]);


  return (
    <div className="min-h-screen bg-background text-foreground p-4 md:p-8"> {/* Use theme background/foreground */}
      <header className="mb-8">
        <h1 className="text-3xl font-bold text-primary flex items-center gap-2">
          <ShieldCheck className="h-8 w-8 text-accent" />
          Node Threat Sentinel
        </h1>
        <p className="text-muted-foreground">Real-time Threat Detection & Analysis PoC</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Activity Log */}
        <Card className="lg:col-span-2 bg-card shadow-lg rounded-lg border border-border"> {/* Theme card, shadow, border */}
          <CardHeader>
            <CardTitle className="flex items-center gap-2 text-card-foreground"> {/* Theme card text */}
              <ActivityLog size={20} /> Activity Log
            </CardTitle>
             <CardDescription className="text-muted-foreground">Stream of system activities (latest first).</CardDescription>
          </CardHeader>
          <CardContent>
             {/* Use ScrollArea for consistent scrollbars */}
             <ScrollArea className="h-[600px] p-4 border border-input rounded-md bg-secondary/30"> {/* Theme border, input, secondary bg */}
                 {activityLog.length === 0 && !isProcessing && <p className="text-muted-foreground italic text-center py-4">No activity logged yet...</p>}
                 {/* Show skeleton loader only when processing the very first item */}
                 {isProcessing && activityLog.length === 0 && (
                    <div className="space-y-3">
                        <Skeleton className="h-16 w-full" />
                        <Skeleton className="h-16 w-full" />
                        <Skeleton className="h-16 w-full" />
                    </div>
                 )}
                 {activityLog.map((activity) => (
                    // Use theme colors for border and background based on type
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
                       {/* Use muted foreground for metadata */}
                       <div className="text-xs mt-1 text-muted-foreground flex flex-wrap gap-x-4 gap-y-1">
                           <span>IP: {activity.sourceIp || 'N/A'}</span>
                           <span>User: {activity.userId || 'N/A'}</span>
                       </div>
                    </div>
                 ))}
             </ScrollArea>
          </CardContent>
        </Card>

        {/* Alerts */}
        <div className="space-y-6">
          {/* Latest Alert Card */}
          <Card className="bg-card shadow-lg rounded-lg border border-border">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-accent">
                <AlertCircle className="h-6 w-6" /> Latest Alert
              </CardTitle>
               <CardDescription className="text-muted-foreground">Details of the most recent suspicious activity detected.</CardDescription>
            </CardHeader>
            <CardContent>
              {latestAlert ? (
                // Use Alert component with destructive variant for consistency
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
