// src/types/activity.ts

export interface Activity {
  id: string;
  timestamp: string; // ISO 8601 format
  type: 'normal' | 'suspicious';
  description: string;
  sourceIp?: string;
  userId?: string;
  system: string; // e.g., 'Firewall', 'Auth Server', 'Database'
  activity: string; // e.g., 'Login Attempt', 'File Access', 'Query Execution'
  data?: Record<string, any>; // Additional context-specific data
}
