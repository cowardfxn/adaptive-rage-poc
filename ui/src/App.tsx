import React from "react";
import { ConfigProvider, theme } from "antd";
import QuestionAnswerForm from "./components/QuestionAnswerForm";
import "./App.css";

const App: React.FC = () => {
  return (
    <ConfigProvider
      theme={{
        algorithm: theme.defaultAlgorithm,
        token: {
          colorPrimary: "#1890ff",
          borderRadius: 8,
        },
      }}
    >
      <QuestionAnswerForm />
    </ConfigProvider>
  );
};

export default App;
