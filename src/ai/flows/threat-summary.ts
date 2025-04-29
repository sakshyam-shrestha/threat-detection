'use server';

/**
 * @fileOverview Summarizes detected threats for administrators using GenAI.
 *
 * - summarizeThreat - A function that summarizes threat information.
 * - ThreatSummaryInput - The input type for the summarizeThreat function.
 * - ThreatSummaryOutput - The return type for the summarizeThreat function.
 */

import {ai} from '@/ai/ai-instance';
import {z} from 'genkit';

const ThreatSummaryInputSchema = z.object({
  threatDescription: z
    .string()
    .describe('Detailed description of the detected threat.'),
});
export type ThreatSummaryInput = z.infer<typeof ThreatSummaryInputSchema>;

const ThreatSummaryOutputSchema = z.object({
  summary: z
    .string()
    .describe('A concise summary of the detected threat, including its nature and potential impact.'),
});
export type ThreatSummaryOutput = z.infer<typeof ThreatSummaryOutputSchema>;

export async function summarizeThreat(input: ThreatSummaryInput): Promise<ThreatSummaryOutput> {
  return threatSummaryFlow(input);
}

const prompt = ai.definePrompt({
  name: 'threatSummaryPrompt',
  input: {
    schema: z.object({
      threatDescription: z
        .string()
        .describe('Detailed description of the detected threat.'),
    }),
  },
  output: {
    schema: z.object({
      summary: z
        .string()
        .describe('A concise summary of the detected threat, including its nature and potential impact.'),
    }),
  },
  prompt: `You are an AI assistant specializing in cybersecurity threat analysis.
  Your task is to provide a summarized description of a detected threat based on the provided details.
  The summary should be concise, focusing on the nature and potential impact of the threat.

  Threat Description: {{{threatDescription}}}
  Summary: `,
});

const threatSummaryFlow = ai.defineFlow<
  typeof ThreatSummaryInputSchema,
  typeof ThreatSummaryOutputSchema
>({
  name: 'threatSummaryFlow',
  inputSchema: ThreatSummaryInputSchema,
  outputSchema: ThreatSummaryOutputSchema,
},
async input => {
  const {output} = await prompt(input);
  return output!;
});

