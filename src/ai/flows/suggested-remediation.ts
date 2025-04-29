// The AI models (Isolation Forest, One-Class SVM, Autoencoders, Random Forest, Gradient Boosting, LSTM, Prophet) will be implemented in Python and integrated with the Node.js app.
// As an administrator, I want the system to suggest remediation steps for detected threats using GenAI, so I can quickly take action to mitigate risks.

'use server';
/**
 * @fileOverview This file defines a Genkit flow that suggests remediation steps for detected threats.
 *
 * - suggestedRemediation - A function that takes threat details as input and returns suggested remediation steps.
 * - SuggestedRemediationInput - The input type for the suggestedRemediation function.
 * - SuggestedRemediationOutput - The return type for the suggestedRemediation function.
 */

import {ai} from '@/ai/ai-instance';
import {z} from 'genkit';

const SuggestedRemediationInputSchema = z.object({
  threatDescription: z.string().describe('A detailed description of the detected threat.'),
  threatCategory: z.string().describe('The category of the threat (e.g., malware, phishing, intrusion).'),
  affectedSystem: z.string().describe('The system or application affected by the threat.'),
});
export type SuggestedRemediationInput = z.infer<typeof SuggestedRemediationInputSchema>;

const SuggestedRemediationOutputSchema = z.object({
  suggestedActions: z
    .array(z.string())
    .describe('A list of suggested actions to remediate the detected threat.'),
  rationale: z.string().describe('A brief rationale for the suggested actions.'),
});
export type SuggestedRemediationOutput = z.infer<typeof SuggestedRemediationOutputSchema>;

export async function suggestedRemediation(input: SuggestedRemediationInput): Promise<SuggestedRemediationOutput> {
  return suggestedRemediationFlow(input);
}

const suggestedRemediationPrompt = ai.definePrompt({
  name: 'suggestedRemediationPrompt',
  input: {
    schema: z.object({
      threatDescription: z.string().describe('A detailed description of the detected threat.'),
      threatCategory: z.string().describe('The category of the threat (e.g., malware, phishing, intrusion).'),
      affectedSystem: z.string().describe('The system or application affected by the threat.'),
    }),
  },
  output: {
    schema: z.object({
      suggestedActions: z
        .array(z.string())
        .describe('A list of suggested actions to remediate the detected threat.'),
      rationale: z.string().describe('A brief rationale for the suggested actions.'),
    }),
  },
  prompt: `You are an expert security analyst providing remediation steps for detected threats.

  Based on the threat description, category, and affected system, suggest a list of actions to remediate the threat and provide a brief rationale for each action.

  Threat Description: {{{threatDescription}}}
  Threat Category: {{{threatCategory}}}
  Affected System: {{{affectedSystem}}}
  \n  Provide output as a valid JSON array of strings.
  `,
});

const suggestedRemediationFlow = ai.defineFlow<
  typeof SuggestedRemediationInputSchema,
  typeof SuggestedRemediationOutputSchema
>({
  name: 'suggestedRemediationFlow',
  inputSchema: SuggestedRemediationInputSchema,
  outputSchema: SuggestedRemediationOutputSchema,
},
async input => {
  const {output} = await suggestedRemediationPrompt(input);
  return output!;
}
);
