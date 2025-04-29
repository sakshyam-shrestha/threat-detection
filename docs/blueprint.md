# **App Name**: Node Threat Sentinel

## Core Features:

- Data Simulation: Simulate data collection to generate fake normal and suspicious activity data.
- Python AI Integration: Leverage Python subprocess calls to execute AI models, using the threatAnalyzer.ts file as a tool to help analyze and categorize threats and behaviors.
- Alert Notification: Implement email alerts exclusively for detected suspicious behavior.

## Style Guidelines:

- Primary color: Dark blue (#1A237E) for a sense of security and stability.
- Secondary color: Light gray (#EEEEEE) for a clean and modern background.
- Accent: Teal (#00ACC1) to highlight important alerts and notifications.
- Clean and structured layout for easy monitoring of system activity.
- Use security-themed icons to represent different data types and alerts.

## Original User Request:
Based on your request, you want a proof-of-concept (PoC) demo that adheres to the architecture described at https://bdoshnkn.manus.space/integration, but with a Node.js application as the primary runtime environment, incorporating a threatAnalyzer.ts TypeScript file that could theoretically run in a browser (though you’ve specified no frontend, so I’ll focus on a backend-only Node.js app). The AI models (Isolation Forest, One-Class SVM, Autoencoders, Random Forest, Gradient Boosting, LSTM, Prophet) will be implemented in Python and integrated with the Node.js app. The system will:

    Use fake data to simulate normal and suspicious behavior.
    Differentiate normal vs. suspicious behavior, sending email notifications only for suspicious behavior.
    Follow the exact structure from the link: Data Collection, AI Model Processing, Threat Analysis, and Alert System layers with all specified components.
    Implement AI models in Python, invoked from Node.js via a subprocess or API.
    Include a threatAnalyzer.ts file in the Node.js app to handle threat analysis logic, designed to be compatible with browser execution (though used server-side here).
  