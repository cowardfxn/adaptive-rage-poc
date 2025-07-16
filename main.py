from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict
from fastapi.responses import PlainTextResponse
import uvicorn

from graph import app as graph_app


app = FastAPI()


# Health check endpoint
@app.get("/ping", response_class=PlainTextResponse)
def ping():
    return "pong"


class QuestionRequest(BaseModel):
    question: str


class GraphResponse(BaseModel):
    generation: str
    state: Dict[str, Any]


@app.post("/api/ask", response_model=GraphResponse)
def ask_question(request: QuestionRequest):
    # return GraphResponse(generation=str("abcdf"), state={"aa": "aabbccdd"})
    try:
        # Run the graph app with the user question
        inputs = {"question": request.question}
        final_state = None
        for output in graph_app.stream(inputs):
            final_state = output
        if not final_state:
            raise HTTPException(status_code=500, detail="No response from graph app.")
        # Find the node with the final generation
        for key, value in final_state.items():
            if isinstance(value, dict) and "generation" in value:
                return GraphResponse(generation=value["generation"], state=value)
            elif key == "generation":
                return GraphResponse(generation=value, state=final_state)
        # Fallback: return the whole state
        return GraphResponse(generation=str(final_state), state=final_state)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
