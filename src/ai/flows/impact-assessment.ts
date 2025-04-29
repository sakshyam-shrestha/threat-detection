// threat-analyzer.ts
'use server';
/**
 * @fileOverview Assesses the potential impact of suspicious behavior using GenAI to prioritize incident response.
 *
 * - assessImpact - A function that assesses the impact of suspicious behavior.
 * - AssessImpactInput - The input type for the assessImpact function.
 * - AssessImpactOutput - The return type for the assessImpact function.
 */

import {ai} from '@/ai/ai-instance';
import {z} from 'genkit';

const AssessImpactInputSchema = z.object({
  behaviorDescription: z
    .string()
    .describe('A description of the suspicious behavior detected.'),
  affectedSystem: z.string().describe('The system affected by the behavior.'),
  potentialVulnerability: z
    .string()
    .describe('The potential vulnerability exploited.'),
});
export type AssessImpactInput = z.infer<typeof AssessImpactInputSchema>;

const AssessImpactOutputSchema = z.object({
  impactScore: z
    .number()
    .describe(
      'A numerical score representing the potential impact (0-100, higher is more severe).' // More specific description
    ),
  impactSummary: z
    .string()
    .describe('A summary of the potential impact of the behavior.'),
  recommendedActions: z
    .string()
    .describe('Recommended actions to mitigate the impact.'),
});
export type AssessImpactOutput = z.infer<typeof AssessImpactOutputSchema>;

export async function assessImpact(input: AssessImpactInput): Promise<AssessImpactOutput> {
  return assessImpactFlow(input);
}

const assessImpactPrompt = ai.definePrompt({
  name: 'assessImpactPrompt',
  input: {
    schema: z.object({
      behaviorDescription: z
        .string()
        .describe('A description of the suspicious behavior detected.'),
      affectedSystem: z.string().describe('The system affected by the behavior.'),
      potentialVulnerability: z
        .string()
        .describe('The potential vulnerability exploited.'),
    }),
  },
  output: {
    schema: z.object({
      impactScore: z
        .number()
        .describe(
          'A numerical score representing the potential impact (0-100, higher is more severe).' // More specific description
        ),
      impactSummary: z
        .string()
        .describe('A summary of the potential impact of the behavior.'),
      recommendedActions: z
        .string()
        .describe('Recommended actions to mitigate the impact.'),
    }),
  },
  prompt: `You are a security expert assessing the impact of suspicious behavior on a system.

  Description of the suspicious behavior: {{{behaviorDescription}}}
  Affected system: {{{affectedSystem}}}
  Potential vulnerability exploited: {{{potentialVulnerability}}}

  Based on this information, assess the potential impact by providing:
  1. An impact score from 0 to 100 (higher is more severe), representing the severity of the impact.
  2. A concise summary of the potential impact of the behavior.
  3. Recommended actions to mitigate the impact.
  Make sure the impactScore is a valid number.
  `,
});

const assessImpactFlow = ai.defineFlow<
  typeof AssessImpactInputSchema,
  typeof AssessImpactOutputSchema
>(
  {
    name: 'assessImpactFlow',
    inputSchema: AssessImpactInputSchema,
    outputSchema: AssessImpactOutputSchema,
  },
  async input => {
    const {output} = await assessImpactPrompt(input);
    return output!;
  }
);
