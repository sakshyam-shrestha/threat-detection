/**
 * @fileOverview Threat Analysis Logic
 * This file contains functions for analyzing activity data to identify potential threats.
 * It's designed to be potentially runnable in different environments (Node.js, Browser),
 * although in this PoC, it's used server-side within the Next.js app.
 *
 * In a real application, this might involve more complex logic, state management,
 * or interactions with external threat intelligence feeds.
 */

import type { Activity } from '@/types/activity';


/**
 * Placeholder for analyzing a single activity event.
 * In a real scenario, this could involve rule-based checks,
 * anomaly detection heuristics, or calls to ML models.
 *
 * For this PoC, the 'suspicious' type is pre-determined during data simulation.
 * This function primarily acts as a placeholder for more complex analysis logic.
 *
 * @param activity The activity data to analyze.
 * @returns An object indicating if the activity is suspicious and potentially other metadata.
 */
export async function analyzeActivity(activity: Activity): Promise<{ isSuspicious: boolean; analysisDetails?: string }> {
  console.log(`[ThreatAnalyzer] Analyzing activity ID: ${activity.id}`);

  // Basic rule example (could be expanded significantly)
  if (activity.description.toLowerCase().includes('drop table') || activity.description.toLowerCase().includes('malicious ip')) {
     console.log(`[ThreatAnalyzer] Activity ${activity.id} flagged as suspicious by basic rule.`);
     return { isSuspicious: true, analysisDetails: "Flagged by basic rule (DROP TABLE or Malicious IP)." };
  }

  // In this PoC, we rely on the pre-assigned 'type' from simulation
  // In a real app, complex logic or ML model output would determine this.
  const isSuspicious = activity.type === 'suspicious';

  if (isSuspicious) {
    console.log(`[ThreatAnalyzer] Activity ${activity.id} confirmed as suspicious.`);
    return { isSuspicious: true, analysisDetails: "Marked as suspicious during simulation." };
  } else {
    console.log(`[ThreatAnalyzer] Activity ${activity.id} deemed normal.`);
    return { isSuspicious: false };
  }
}


// Example of how this *could* be used if run standalone or in a different context
// This part is not directly used by the Next.js page component in this PoC.
async function runStandaloneAnalysis() {
  const sampleSuspiciousActivity: Activity = {
    id: 'standalone-test-123',
    timestamp: new Date().toISOString(),
    type: 'suspicious',
    description: 'Multiple failed login attempts followed by successful login from unusual IP.',
    sourceIp: '172.16.254.1',
    userId: 'admin_user',
    system: 'Authentication Server',
    activity: 'Anomalous Login Pattern',
    data: { attempts: 5, location: 'Unknown' },
  };

  const initialCheck = await analyzeActivity(sampleSuspiciousActivity);
  console.log("Initial Check Result:", initialCheck);

}

// Uncomment the line below to run the standalone example if needed (e.g., `node src/ai/threatAnalyzer.js`)
// runStandaloneAnalysis();
