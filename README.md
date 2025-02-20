# 🧬 AI Co-Scientist System

> 🤖 An autonomous research system for scientific discovery and hypothesis generation

## 🌟 Overview

The AI Co-Scientist is a sophisticated multi-agent system designed to autonomously conduct scientific research, generate and evaluate hypotheses, and synthesize findings. The system employs a collaborative network of specialized agents, each responsible for different aspects of the research process.

## 🚀 Features

- 🧪 Autonomous hypothesis generation and evaluation
- 📊 Systematic research planning and execution
- 🔍 Advanced hypothesis clustering and similarity analysis
- 🏆 Tournament-based hypothesis ranking
- 🔄 Continuous hypothesis refinement and evolution
- 📈 Comprehensive research synthesis and meta-review

## 🏗️ System Architecture

The system consists of seven specialized agents:

1. 👨‍💼 **Supervisor Agent**: Orchestrates the research process and manages agent interactions
2. 💡 **Generation Agent**: Creates novel research hypotheses based on goals and context
3. 🔬 **Reflection Agent**: Evaluates hypotheses through multiple review types
4. 🎯 **Ranking Agent**: Conducts tournament-style comparisons between hypotheses
5. 📈 **Evolution Agent**: Refines and improves promising hypotheses
6. 🔗 **Proximity Agent**: Analyzes relationships and clusters similar hypotheses
7. 📊 **Meta-Review Agent**: Synthesizes findings and generates research overviews

## ⚙️ Current Progress

### Core Functionality
- [x] Basic agent framework and communication
- [x] Research goal setting and planning
- [x] Hypothesis generation with LLM integration
- [x] Multi-stage hypothesis review system
- [x] Tournament-based hypothesis ranking
- [x] Hypothesis refinement and evolution
- [x] Clustering and similarity analysis
- [x] Research synthesis and meta-review

### Output Handling
- [x] Proper JSON formatting for LLM outputs
- [x] Robust error handling for malformed responses
- [x] Default fallback values for all agents
- [x] Consistent data structures across agents

### Research Cycle
- [x] Single research cycle completion
- [ ] Multiple cycle management
- [ ] Cycle termination conditions
- [ ] Progress tracking across cycles

### Advanced Features
- [ ] Long-term memory and knowledge accumulation
- [ ] Cross-cycle hypothesis evolution
- [ ] Dynamic strategy adaptation
- [ ] Automated experiment design
- [ ] External knowledge integration
- [ ] Collaborative research capabilities

### Quality and Validation
- [ ] Hypothesis quality metrics
- [ ] Research impact assessment
- [ ] Validation frameworks
- [ ] Performance benchmarks
- [ ] Testing suite

## 🛠️ Technical Requirements

- Python 3.8+
- LangChain
- Pydantic
- NumPy
- scikit-learn
- Rich (for console output)

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/ai-co-scientist.git

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage

```python
from agents.co_scientist.main import AICoScientist
from langchain_ollama import ChatOllama

# Initialize the system
llm = ChatOllama(model="deepscaler", format="json")
scientist = AICoScientist(llm)

# Define research goal
research_goal = {
    "goal": "Investigate potential drug repurposing candidates for treating AML",
    "domain": "drug_repurposing",
    "constraints": ["Focus on FDA-approved drugs"],
    "preferences": {"prioritize_novel_mechanisms": True}
}

# Run research process
results = await scientist.run(research_goal)
```

## 📄 License

The Unlicense

This is free and unencumbered software released into the public domain.

Anyone is free to copy, modify, publish, use, compile, sell, or distribute this software, either in source code form or as a compiled binary, for any purpose, commercial or non-commercial, and by any means.

In jurisdictions that recognize copyright laws, the author or authors of this software dedicate any and all copyright interest in the software to the public domain. We make this dedication for the benefit of the public at large and to the detriment of our heirs and successors. We intend this dedication to be an overt act of relinquishment in perpetuity of all present and future rights to this software under copyright law.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

For more information, please refer to <http://unlicense.org/>

## 🤝 Contributing

Contributions are welcome! Feel free to:

- 🐛 Report bugs
- 💡 Suggest features
- 🔧 Submit pull requests
- 📚 Improve documentation

## 📊 Project Status

The system is currently in active development with basic functionality implemented. The core research cycle is working, and agents can successfully generate, evaluate, and refine hypotheses. Work is ongoing to implement advanced features and improve robustness.

### Next Steps

1. 🔄 Implement proper cycle termination conditions
2. 💾 Add persistent memory across research cycles
3. 🔬 Develop automated experiment design
4. 🤝 Enable multi-agent collaboration
5. 📈 Implement comprehensive metrics and validation 