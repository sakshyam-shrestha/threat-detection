'use client';

import React, { useState, useEffect, useCallback } from 'react';
import { Card, CardHeader, CardTitle, CardContent, CardDescription, CardFooter } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Alert, AlertTitle, AlertDescription } from '@/components/ui/alert';
import { ActivityLog } from '@/components/activity-log';
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
  ];

  const descriptionIndex = Math.floor(Math.random() * descriptions.length);

  return {
    id: crypto.randomUUID(),
    timestamp: timestamp.toISOString(),
    type: isSuspicious ? 'suspicious' : 'normal',
    description: descriptions[descriptionIndex],
    sourceIp: `192.168.${Math.floor(Math.random() * 255)}.${Math.floor(Math.random() * 255)}`,
    userId: isSuspicious && Math.random() > 0.5 ? 'unknown' : `user_${Math.floor(Math.random() * 1000)}`,
    system: systems[Math.floor(Math.random() * systems.length)],
    activity: activities[Math.floor(Math.random() * activities.length)],
    data: { detail: `Detail about ${activities[Math.floor(Math.random() * activities.length)]}` }
  };
};

const getIconForSystem = (system: string) => {
  switch (system) {
    case 'Auth Server': return <Server className="h-5 w-5 text-primary" />;
    case 'File Server': return <Database className="h-5 w-5 text-green-600" />;
    case 'Web App': return <Cpu className="h-5 w-5 text-blue-600" />;
    case 'Database Cluster': return <Database className="h-5 w-5 text-purple-600" />;
    case 'Firewall': return <ShieldCheck className="h-5 w-5 text-red-600" />;
    case 'API Gateway': return <Network className="h-5 w-5 text-yellow-600" />;
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
    setIsProcessing(true);
    console.log(`Processing data: ${data.id} - ${data.description}`);

    // Simulate AI model processing (replace with actual Python call)
    await new Promise(resolve => setTimeout(resolve, 500)); // Simulate async call

    const isSuspicious = data.type === 'suspicious'; // Using the simulated type

    setActivityLog(prevLog => [data, ...prevLog.slice(0, 99)]); // Keep log size manageable

    if (isSuspicious) {
      console.log(`Suspicious activity detected: ${data.description}`);
      setLatestAlert(data);
      setIsAnalyzing(true);
      setAiAnalysis(null); // Reset previous analysis

      try {
        // Simulate sending email alert
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

        // Trigger GenAI flows
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
          variant: "default",
        });

      } catch (error) {
        console.error("Error during alert or AI processing:", error);
        toast({
          title: "Processing Error",
          description: "Failed to send alert or perform AI analysis.",
          variant: "destructive",
        });
        setAiAnalysis({ // Provide fallback error state
          summary: "Error during AI analysis.",
          impact: -1,
          remediation: ["Manual investigation required."],
        });
      } finally {
        setIsAnalyzing(false);
      }
    } else {
       // Optionally clear the alert if the latest activity is normal
       // setLatestAlert(null);
       // setAiAnalysis(null);
    }

    setIsProcessing(false);
  }, [toast]); // Added toast dependency

  // Simulate receiving new data every few seconds
  useEffect(() => {
    const intervalId = setInterval(() => {
      if (!isProcessing) {
        const newData = generateFakeData();
        processData(newData);
      }
    }, 3000); // Generate data every 3 seconds

    // Initial data load
    for (let i = 0; i < 5; i++) {
       processData(generateFakeData());
    }


    return () => clearInterval(intervalId); // Cleanup on unmount
  }, [processData, isProcessing]); // Added isProcessing to dependency array


  return (
    <div className="min-h-screen bg-secondary p-4 md:p-8">
      <header className="mb-8">
        <h1 className="text-3xl font-bold text-primary flex items-center gap-2">
          <ShieldCheck className="h-8 w-8 text-accent" />
          Node Threat Sentinel
        </h1>
        <p className="text-muted-foreground">Real-time Threat Detection & Analysis PoC</p>
      </header>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Activity Log */}
        <Card className="lg:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <ActivityLog size={20} /> Activity Log
            </CardTitle>
             <CardDescription>Stream of system activities (latest first).</CardDescription>
          </CardHeader>
          <CardContent>
             <ScrollArea className="h-[600px] p-4 border rounded-md bg-background">
                 {activityLog.length === 0 && !isProcessing && <p>No activity yet...</p>}
                 {isProcessing && activityLog.length === 0 && <Skeleton className="h-10 w-full mb-2" count={5} />}
                 {activityLog.map((activity) => (
                    <div key={activity.id} className={`mb-3 p-3 rounded-md border ${activity.type === 'suspicious' ? 'border-destructive bg-destructive/10' : 'border-border'}`}>
                      <div className="flex justify-between items-center mb-1">
                         <div className="flex items-center gap-2">
                            {getIconForSystem(activity.system)}
                           <span className={`font-semibold ${activity.type === 'suspicious' ? 'text-destructive' : 'text-foreground'}`}>
                             {activity.activity} on {activity.system}
                           </span>
                         </div>
                         <span className="text-xs text-muted-foreground">
                           {new Date(activity.timestamp).toLocaleString()}
                         </span>
                      </div>
                       <p className="text-sm text-muted-foreground truncate" title={activity.description}>{activity.description}</p>
                       <div className="text-xs mt-1 text-muted-foreground/80 flex gap-4">
                           <span>IP: {activity.sourceIp}</span>
                           <span>User: {activity.userId}</span>
                       </div>
                    </div>
                 ))}
             </ScrollArea>
          </CardContent>
        </Card>

        {/* Alerts and AI Analysis */}
        <div className="space-y-6">
          <Card className="bg-card shadow-md">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-accent">
                <AlertCircle className="h-6 w-6" /> Latest Alert
              </CardTitle>
               <CardDescription>Details of the most recent suspicious activity detected.</CardDescription>
            </CardHeader>
            <CardContent>
              {latestAlert ? (
                <Alert variant="destructive" className="bg-destructive/10 border-destructive/50">
                  <AlertCircle className="h-4 w-4" />
                  <AlertTitle>Suspicious Activity Detected!</AlertTitle>
                  <AlertDescription>
                    <p><strong>Time:</strong> {new Date(latestAlert.timestamp).toLocaleString()}</p>
                    <p><strong>Description:</strong> {latestAlert.description}</p>
                    <p><strong>System:</strong> {latestAlert.system}</p>
                    <p><strong>Source IP:</strong> {latestAlert.sourceIp}</p>
                    <p><strong>User:</strong> {latestAlert.userId}</p>
                  </AlertDescription>
                </Alert>
              ) : (
                <p className="text-muted-foreground">No suspicious activity detected recently.</p>
              )}
            </CardContent>
          </Card>

          <Card className="bg-card shadow-md">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                 <BarChart className="h-6 w-6 text-primary"/> AI Analysis
              </CardTitle>
              <CardDescription>GenAI insights on the latest alert.</CardDescription>
            </CardHeader>
            <CardContent>
              {isAnalyzing && (
                 <div className="space-y-4">
                    <Skeleton className="h-4 w-3/4" />
                    <Skeleton className="h-4 w-1/2" />
                    <Skeleton className="h-4 w-full" />
                    <Skeleton className="h-4 w-5/6" />
                 </div>
              )}
              {!isAnalyzing && aiAnalysis && (
                <div className="space-y-3">
                   <div>
                      <h4 className="font-semibold">Impact Score:</h4>
                      <p className={`text-lg font-bold ${aiAnalysis.impact > 70 ? 'text-destructive' : aiAnalysis.impact > 40 ? 'text-yellow-600' : 'text-green-600'}`}>
                         {aiAnalysis.impact >= 0 ? `${aiAnalysis.impact} / 100` : 'Error'}
                      </p>
                   </div>
                  <div>
                    <h4 className="font-semibold">Threat Summary:</h4>
                    <p className="text-sm text-muted-foreground">{aiAnalysis.summary}</p>
                  </div>
                  <div>
                    <h4 className="font-semibold">Suggested Remediation:</h4>
                    <ul className="list-disc list-inside text-sm text-muted-foreground space-y-1">
                      {aiAnalysis.remediation.map((step, index) => (
                        <li key={index}>{step}</li>
                      ))}
                    </ul>
                  </div>
                </div>
              )}
              {!isAnalyzing && !aiAnalysis && latestAlert && (
                 <p className="text-muted-foreground">Waiting for AI analysis results...</p>
              )}
               {!isAnalyzing && !aiAnalysis && !latestAlert && (
                 <p className="text-muted-foreground">No alert to analyze.</p>
               )}
            </CardContent>
             <CardFooter>
                 <Button
                     onClick={() => latestAlert && processData(latestAlert)} // Re-analyze the latest alert
                     disabled={!latestAlert || isAnalyzing || isProcessing}
                     size="sm"
                     variant="outline"
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
