import axios from "axios";
import type { QuestionRequest, GraphResponse } from "../types/api";

const API_PREFIX = "/api";

export const askQuestion = async (
  request: QuestionRequest
): Promise<GraphResponse> => {
  const response = await axios.post<GraphResponse>(
    `${API_PREFIX}/ask`,
    request
  );
  return response.data;
};
