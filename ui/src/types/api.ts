export interface QuestionRequest {
  question: string;
}

export interface GraphResponse {
  generation: string;
  state: Record<string, any>;
}
