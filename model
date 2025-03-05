import os
from typing import List, Dict, Any
import asyncio

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from tavily import TavilyClient

# Configuration and Environment Setup
class ResearchConfig:
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    TAVILY_API_KEY = os.getenv('TAVILY_API_KEY')
    DEFAULT_MODEL = "gpt-4-turbo"
    RESEARCH_DEPTH = 5  # Number of search iterations
    MAX_SOURCES = 10  # Maximum number of sources to collect

class ResearchState:
    """
    Defines the state management for the research workflow
    """
    def __init__(self):
        self.query: str = ""
        self.initial_search_results: List[Dict] = []
        self.refined_search_results: List[Dict] = []
        self.research_summary: str = ""
        self.final_answer: str = ""
        self.sources: List[str] = []

class ResearchAgent:
    def __init__(self):
        self.tavily_client = TavilyClient(api_key=ResearchConfig.TAVILY_API_KEY)
        self.llm = ChatOpenAI(
            model=ResearchConfig.DEFAULT_MODEL, 
            temperature=0.3
        )

    async def initial_search(self, state: ResearchState) -> ResearchState:
        """
        Perform initial web search using Tavily
        """
        search_results = self.tavily_client.search(
            query=state.query, 
            max_results=ResearchConfig.MAX_SOURCES
        )
        
        state.initial_search_results = search_results.get('results', [])
        state.sources = [result['url'] for result in state.initial_search_results]
        
        return state

    async def refine_search(self, state: ResearchState) -> ResearchState:
        """
        Refine search based on initial results
        """
        refine_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert researcher. Analyze the initial search results and generate a more targeted search query."),
            ("human", "Initial Query: {query}\nInitial Results: {results}")
        ])
        
        refine_chain = refine_prompt | self.llm | StrOutputParser()
        
        refined_query = await refine_chain.ainvoke({
            "query": state.query,
            "results": str(state.initial_search_results)
        })
        
        refined_search = self.tavily_client.search(
            query=refined_query, 
            max_results=ResearchConfig.MAX_SOURCES
        )
        
        state.refined_search_results = refined_search.get('results', [])
        state.sources.extend([result['url'] for result in state.refined_search_results])
        
        return state

    async def synthesize_research(self, state: ResearchState) -> ResearchState:
        """
        Synthesize research findings into a coherent summary
        """
        synthesis_prompt = ChatPromptTemplate.from_messages([
            ("system", "Synthesize the research findings into a comprehensive and well-structured summary."),
            ("human", "Research Query: {query}\nSearch Results: {results}")
        ])
        
        synthesis_chain = synthesis_prompt | self.llm | StrOutputParser()
        
        state.research_summary = await synthesis_chain.ainvoke({
            "query": state.query,
            "results": str(state.initial_search_results + state.refined_search_results)
        })
        
        return state

class AnswerDraftAgent:
    def __init__(self):
        self.llm = ChatOpenAI(
            model=ResearchConfig.DEFAULT_MODEL, 
            temperature=0.2
        )

    async def draft_answer(self, state: ResearchState) -> ResearchState:
        """
        Draft a comprehensive answer based on research summary
        """
        draft_prompt = ChatPromptTemplate.from_messages([
            ("system", "Draft a comprehensive and well-structured answer based on the research summary."),
            ("human", "Research Query: {query}\nResearch Summary: {summary}")
        ])
        
        draft_chain = draft_prompt | self.llm | StrOutputParser()
        
        state.final_answer = await draft_chain.ainvoke({
            "query": state.query,
            "summary": state.research_summary
        })
        
        return state

def create_research_workflow():
    """
    Create the LangGraph workflow for research
    """
    workflow = StateGraph(ResearchState)
    
    research_agent = ResearchAgent()
    answer_agent = AnswerDraftAgent()
    
    # Add nodes
    workflow.add_node("initial_search", research_agent.initial_search)
    workflow.add_node("refine_search", research_agent.refine_search)
    workflow.add_node("synthesize_research", research_agent.synthesize_research)
    workflow.add_node("draft_answer", answer_agent.draft_answer)
    
    # Define workflow edges
    workflow.set_entry_point("initial_search")
    workflow.add_edge("initial_search", "refine_search")
    workflow.add_edge("refine_search", "synthesize_research")
    workflow.add_edge("synthesize_research", "draft_answer")
    workflow.add_edge("draft_answer", END)
    
    return workflow.compile()

async def run_deep_research(query: str):
    """
    Execute the deep research workflow
    """
    workflow = create_research_workflow()
    
    initial_state = ResearchState()
    initial_state.query = query
    
    final_state = await workflow.ainvoke(initial_state)
    
    return {
        "query": final_state.query,
        "sources": final_state.sources,
        "final_answer": final_state.final_answer
    }

# Example Usage
async def main():
    research_result = await run_deep_research(
        "What are the latest advancements in AI language models?"
    )
    print(research_result['final_answer'])
    print("\nSources:", research_result['sources'])

if __name__ == "__main__":
    asyncio.run(main())
