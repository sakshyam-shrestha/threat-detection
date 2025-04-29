'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert';
import { ActivityLog } from '@/components/activity-log'; // Custom icon component
import { useToast } from "@/hooks/use-toast";
import { sendEmail } from '@/services/email';
import { assessImpact } from '@/ai/flows/impact-assessment';
import { summarizeThreat } from '@/ai/flows/threat-summary';
import { suggestedRemediation } from '@/ai/flows/suggested-remediation';
import type { Activity } from '@/types/activity';
import { AlertCircle, ShieldCheck, Cpu, Network, Server, Database, BarChart, Terminal } from 'lucide-react';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Skeleton } from '@/components/ui/skeleton';

// Simulate data generation (replace with actual data source)
const generateFakeData = (): Activity => {
  // Make suspicious activity less frequent to simulate more realistic threat scenarios
  const isSuspicious = Math.random() < 0.1; // 10% chance of suspicious activity
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

  if (generatedDescription.includes('DROP TABLE') || generatedDescription.includes('malicious IP') || generatedDescription.includes('Disabled firewall')) {
    generatedType = 'suspicious';
  } else if (!isSuspicious && Math.random() < 0.05) { // Occasionally mark non-obvious things as suspicious
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
  const [aiAnalysis, setAiAnalysis] = useState<{ summary: string; impact: number; remediation: string[] } | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const { toast } = useToast();

  const processData = useCallback(async (data: Activity) => {
    if (isProcessing) return; // Prevent concurrent processing

    setIsProcessing(true);
    console.log(`Processing data: ${data.id} - Type: ${data.type} - ${data.description}`);

    // Simulate AI model processing (replace with actual Python call if needed)
    // Keeping a short delay to simulate some backend work without blocking UI too long
    await new Promise(resolve => setTimeout(resolve, 200));

    // Update activity log state immutably
    setActivityLog(prevLog => [data, ...prevLog.slice(0, 99)]); // Keep log size manageable

    if (data.type === 'suspicious') {
      console.log(`Suspicious activity detected: ${data.description}`);
      setLatestAlert(data);
      setIsAnalyzing(true);
      setAiAnalysis(null); // Reset previous analysis

      try {
        // Simulate sending email alert FIRST (critical step)
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
          variant: "default", // Use default variant for info
        });

        // Trigger GenAI flows AFTER alert is sent
        const [summaryResult, impactResult, remediationResult] = await Promise.all([
          summarizeThreat({ threatDescription: data.description }),
          assessImpact({
            behaviorDescription: data.description,
            affectedSystem: data.system,
            potentialVulnerability: 'Unknown/Simulated', // Provide context if available
          }),
          suggestedRemediation({
            threatDescription: data.description,
            threatCategory: data.activity, // Use activity type as category for simplicity
            affectedSystem: data.system,
          }),
        ]);

        setAiAnalysis({
          summary: summaryResult.summary,
          impact: impactResult.impactScore,
          remediation: remediationResult.suggestedActions,
        });

        toast({
          title: "AI Analysis Complete",
          description: "Threat summary, impact assessment, and remediation steps generated.",
          variant: "default", // Use default variant
        });

      } catch (error) {
        console.error("Error during alert or AI processing:", error);
        toast({
          title: "Processing Error",
          description: `Failed to process alert for ${data.id}. Check console.`, // More specific error
          variant: "destructive",
        });
        setAiAnalysis({ // Provide fallback error state
          summary: "Error during AI analysis.",
          impact: -1, // Indicate error state clearly
          remediation: ["Manual investigation required due to processing error."],
        });
      } finally {
        setIsAnalyzing(false);
      }
    }
    // No need for 'else' block to clear alerts if requirement is to keep the last alert displayed

    setIsProcessing(false);
  }, [toast, isProcessing]); // Added isProcessing dependency

  // Simulate receiving new data at a slower, more manageable pace
  useEffect(() => {
    const intervalId = setInterval(() => {
      // No need to check isProcessing here, the check is inside processData
      const newData = generateFakeData();
      processData(newData);
    }, 15000); // Generate data every 15 seconds

    // Load only one initial data point to avoid overwhelming the view at start
    const initialData = generateFakeData();
    processData(initialData);


    return () => clearInterval(intervalId); // Cleanup on unmount
  }, [processData]); // processData depends on isProcessing, so this effect reruns safely


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
                 {/* Show skeleton loaders more appropriately */}
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

        {/* Alerts and AI Analysis */}
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

          {/* AI Analysis Card */}
          <Card className="bg-card shadow-lg rounded-lg border border-border">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-card-foreground">
                 <BarChart className="h-6 w-6 text-primary"/> AI Analysis
              </CardTitle>
              <CardDescription className="text-muted-foreground">GenAI insights on the latest alert.</CardDescription>
            </CardHeader>
            <CardContent className="min-h-[150px]"> {/* Ensure minimum height */}
              {isAnalyzing && (
                 <div className="space-y-4 pt-2">
                    <Skeleton className="h-4 w-3/4" />
                     <Skeleton className="h-4 w-1/2" />
                     <Skeleton className="h-4 w-full" />
                     <Skeleton className="h-4 w-5/6" />
                     <Skeleton className="h-4 w-full" />
                 </div>
              )}
              {!isAnalyzing && aiAnalysis && (
                <div className="space-y-4">
                   <div>
                      <h4 className="font-semibold text-foreground mb-1">Impact Score:</h4>
                      {/* Use theme colors based on score */}
                      <p className={`text-xl font-bold ${aiAnalysis.impact === -1 ? 'text-muted-foreground' : aiAnalysis.impact > 70 ? 'text-destructive' : aiAnalysis.impact > 40 ? 'text-yellow-600' : 'text-green-600'}`}>
                         {aiAnalysis.impact >= 0 ? `${aiAnalysis.impact} / 100` : 'Analysis Error'}
                      </p>
                   </div>
                  <div>
                    <h4 className="font-semibold text-foreground mb-1">Threat Summary:</h4>
                    <p className="text-sm text-muted-foreground">{aiAnalysis.summary}</p>
                  </div>
                  <div>
                    <h4 className="font-semibold text-foreground mb-1">Suggested Remediation:</h4>
                    <ul className="list-disc list-inside text-sm text-muted-foreground space-y-1 pl-2">
                      {aiAnalysis.remediation.map((step, index) => (
                        <li key={index}>{step}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
              {/* Clearer states when no analysis is available */}
              {!isAnalyzing && !aiAnalysis && latestAlert && (
                 <p className="text-muted-foreground italic text-center py-4">Waiting for AI analysis results...</p>
              )}
               {!isAnalyzing && !aiAnalysis && !latestAlert && (
                 <p className="text-muted-foreground italic text-center py-4">No alert detected to analyze.</p>
               )}
            </CardContent>
             <CardFooter>
                 {/* Button styling using theme variants */}
                 <Button
                     onClick={() => latestAlert && processData(latestAlert)} // Re-analyze the latest alert
                     disabled={!latestAlert || isAnalyzing || isProcessing}
                     size="sm"
                     variant="outline" // Use outline variant
                   >
                     {isAnalyzing ? 'Analyzing...' : 'Re-Analyze Latest Alert'}
                 </Button>
             </CardFooter>
          </Card>
        </div>
      </div>
    </div>
  );
}
