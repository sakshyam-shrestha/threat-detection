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
import { assessImpact, AssessImpactInput, AssessImpactOutput } from './flows/impact-assessment';
import { summarizeThreat, ThreatSummaryInput, ThreatSummaryOutput } from './flows/threat-summary';
import { suggestedRemediation, SuggestedRemediationInput, SuggestedRemediationOutput } from './flows/suggested-remediation';

/**
 * Placeholder for analyzing a single activity event.
 * In a real scenario, this could involve rule-based checks,
 * anomaly detection heuristics, or calls to ML models.
 *
 * For this PoC, the 'suspicious' type is pre-determined during data simulation.
 * This function primarily acts as a bridge to the GenAI flows.
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


/**
 * Orchestrates the GenAI analysis for a suspicious activity.
 *
 * @param activity The suspicious activity data.
 * @returns A promise resolving to an object containing summary, impact, and remediation suggestions.
 */
export async function performFullAnalysis(activity: Activity): Promise<{
  summary: ThreatSummaryOutput;
  impact: AssessImpactOutput;
  remediation: SuggestedRemediationOutput;
}> {
  if (activity.type !== 'suspicious') {
    throw new Error("Full analysis should only be performed on suspicious activities.");
  }

  console.log(`[ThreatAnalyzer] Performing full GenAI analysis for activity ID: ${activity.id}`);

  const summaryInput: ThreatSummaryInput = {
    threatDescription: activity.description,
  };

  const impactInput: AssessImpactInput = {
    behaviorDescription: activity.description,
    affectedSystem: activity.system,
    // In a real app, identify potential vulnerability if possible
    potentialVulnerability: `Simulated vulnerability related to ${activity.activity}`,
  };

  const remediationInput: SuggestedRemediationInput = {
    threatDescription: activity.description,
    threatCategory: activity.activity, // Using activity as category for PoC
    affectedSystem: activity.system,
  };

  try {
    // Run GenAI flows in parallel
    const [summaryResult, impactResult, remediationResult] = await Promise.all([
      summarizeThreat(summaryInput),
      assessImpact(impactInput),
      suggestedRemediation(remediationInput),
    ]);

     console.log(`[ThreatAnalyzer] GenAI analysis successful for activity ID: ${activity.id}`);
    return {
      summary: summaryResult,
      impact: impactResult,
      remediation: remediationResult,
    };
  } catch (error) {
    console.error(`[ThreatAnalyzer] Error during GenAI analysis for activity ID ${activity.id}:`, error);
    // Provide a fallback or re-throw depending on desired error handling
     throw new Error(`GenAI analysis failed for activity ${activity.id}.`);
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

  if (initialCheck.isSuspicious) {
    try {
        const fullAnalysisResult = await performFullAnalysis(sampleSuspiciousActivity);
        console.log("\nFull GenAI Analysis Result:");
        console.log("---------------------------");
        console.log("Summary:", fullAnalysisResult.summary.summary);
        console.log("Impact Score:", fullAnalysisResult.impact.impactScore);
        console.log("Impact Summary:", fullAnalysisResult.impact.impactSummary);
        console.log("Recommended Actions:", fullAnalysisResult.impact.recommendedActions);
        console.log("Suggested Remediation:", fullAnalysisResult.remediation.suggestedActions);
        console.log("Remediation Rationale:", fullAnalysisResult.remediation.rationale);
        console.log("---------------------------");
    } catch (error) {
        console.error("Standalone analysis failed:", error)
    }
  }
}

// Uncomment the line below to run the standalone example if needed (e.g., `node src/ai/threatAnalyzer.js`)
// runStandaloneAnalysis();
