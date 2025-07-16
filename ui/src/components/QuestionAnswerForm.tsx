import React, { useState } from "react";
import { Input, Button, Typography, Spin, Card, Space } from "antd";
import type { GraphResponse } from "../types/api";
import { askQuestion } from "../services/api";

const { TextArea } = Input;
const { Title, Paragraph } = Typography;

const QuestionAnswerForm: React.FC = () => {
  const [question, setQuestion] = useState("");
  const [answer, setAnswer] = useState<GraphResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    if (!question.trim()) return;

    setLoading(true);
    setError(null);
    try {
      const response = await askQuestion({ question });
      setAnswer(response);
    } catch (err) {
      setError(err instanceof Error ? err.message : "An error occurred");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div style={{ width: "50%", margin: "40px auto", padding: "0 20px" }}>
      <Card>
        <Title level={2} style={{ textAlign: "center", marginBottom: "24px" }}>
          AI Assistant
        </Title>

        <Space direction="vertical" size="large" style={{ width: "100%" }}>
          <div>
            <TextArea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              placeholder="Enter your question..."
              autoSize={{ minRows: 3, maxRows: 6 }}
              style={{ marginBottom: "16px", width: "100%" }}
            />
            <Button
              type="primary"
              onClick={handleSubmit}
              loading={loading}
              block
            >
              Ask Question
            </Button>
          </div>

          {loading && (
            <div style={{ textAlign: "center", padding: "20px" }}>
              <Spin size="large" />
            </div>
          )}

          {error && <Paragraph type="danger">{error}</Paragraph>}

          {answer && (
            <div>
              <Title level={4}>Answer:</Title>
              <Paragraph style={{ whiteSpace: "pre-wrap" }}>
                {answer.generation}
              </Paragraph>
            </div>
          )}
        </Space>
      </Card>
    </div>
  );
};

export default QuestionAnswerForm;
