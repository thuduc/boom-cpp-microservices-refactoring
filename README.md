## GenAI Proof of Concept: refactor existing C++ codebase into microservices

The purpose of this proof of concept is to find out if an LLM can take an existing complex codebase and refactor it into microservices. The codebase we will be using is the BOOM C++ library for Bayesian modeling: https://github.com/steve-the-bayesian/BOOM

### LLM & AI Tool
* LLM used: Claude Opus 4 (best coding LLM) - https://www.anthropic.com/claude/opus
* AI tool used: Claude Code (best coding CLI due to its integration with Clause 4 LLMs) - https://www.anthropic.com/claude-code

### Conversion Process: 
* Step 1 - use Claude Code (together with Opus 4 LLM) to analyze an existing project's codebase, then ask it to put together a comprehensive refactoring plan to refactor into microservices.
* Step 2 - developer verifies the refactoring plan and modifies the plan as needed. Developer could use Claude Code and iterate through this process. UI mocks could be provided, if we need the UI to look a certain way.
* Step 3 - use this refactoring plan (see [MICROSERVICES_REFACTORING_PLAN.md](MICROSERVICES_REFACTORING_PLAN.md)) in Claude Code (together with Claude Opus 4 LLM) to implement all phases in the plan.

### Conversion Results
* The refactoring effort took Claude Code about 5 hours to complete
* The original codebase was refactored into 10 microservices, each with its own Dockerfile. These 10 microservices reside under microservices/ folder.

  1. ✅ rng-service (Random Number Generation)
  2. ✅ linalg-service (Linear Algebra)
  3. ✅ optimization-service
  4. ✅ mcmc-service (MCMC Sampling)
  5. ✅ statistical-models-service
  6. ✅ glm-service (GLM)
  7. ✅ time-series-service
  8. ✅ mixture-models-service
  9. ✅ special-functions-service
  10. ✅ statistical-utilities-service

## All prompts issued to Claude Code
The complete list of prompts issued to Clause Code is listed below (note that due to complexity of the codebase, we converted 2 services at a time. This allowed us to stop and evaludated the services before continuing the):

> we're planning to refactor the existing codebase into modular microservices. Each microservice will be dockerized and run in its own container. Each microservice will exist in a separate git repo. You're an expert in microservices architecture, come up with a design and a plan on how to refactor the current codebase to support this effort. Save this plan under MICROSERVICES_REFACTORING_PLAN.md

> Go ahead and implement the first 2 microservices in @MICROSERVICES_REFACTORING_PLAN.md . Put all microservices git repos under microservices directory. Make sure each microsservice has sufficient unit test coverage.

> Implement the next 2 microservices

> Implement the next 2 microservices

> Implement the next 2 microservices

> Implement the last 2 microservices
