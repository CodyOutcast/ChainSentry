# ChainSentry Project Information Bank

## 1. Project Goal
ChainSentry is a pre-signature transaction risk assistant for blockchain users. Its purpose is to help users understand what a transaction will do, why it is risky, and what they should do before signing.

The project is not trying to build a full blockchain security platform. The goal is to build a focused prototype that combines a small number of useful risk checks with short, understandable explanations.
The final project direction is to back those checks with a graph-based trained predictor model rather than stopping at a purely rule-based prototype.

## 2. Core Problem
Most wallet confirmation screens show technical data such as addresses, calldata, gas fees, and token values. That information is often not enough for ordinary users to understand the real consequence of signing.

This creates three practical user needs:

• What will this transaction do?
• Why is it risky?
• What should I do next?

Current tools usually solve only part of this problem. Some detect risk but do not explain it clearly. Some simulate outcomes but do not present the results in a useful way for quick decision-making. The main project challenge is therefore not only detection accuracy, but also explanation quality.

## 3. Main Project Question
How can a wallet-connected system detect a small set of common high-risk blockchain transaction patterns and explain them in a way that helps users make better signing decisions?

## 4. Target Outcome
By the end of the project, the team will deliver a working prototype that:

• Accepts a pending transaction before signature.
• Runs a small set of risk checks.
• Produces a short risk card in plain language.
• Gives a clear recommendation such as proceed, inspect further, or reject.
• Can be evaluated with example scenarios or lightweight user testing.

## 5. Target Users
The primary users are non-expert or intermediate blockchain users who can use a wallet but do not understand low-level transaction details.

The system is designed for users who:

• Recognize basic wallet actions such as transfer, swap, and token approval.
• Do not read smart contract code.
• Need fast guidance at signing time.
• Ignore warnings that are too long or too technical.

## 6. Project Scope
The prototype stays narrow. It covers a few risk scenarios that are common, high impact, and easy to explain.

### Included Scope
• Unlimited or unusually large token approvals.
• Transactions involving suspicious or flagged contract addresses.
• Transactions where simulation shows a result that surprises a user, such as unexpected asset movement or privilege escalation.
• A graph-based transaction-risk predictor that models addresses, contracts, tokens, and token flows as a local heterogeneous graph.

### Excluded Scope
• Full static analysis of arbitrary smart contracts.
• Web-scale dynamic graph systems or research-heavy graph models that cannot be trained and evaluated within project scope.
• Large-scale on-chain reputation systems.
• On-chain storage of risk summaries.
• Production wallet deployment across multiple chains.

These items are future extensions and are not part of the project deliverable.

## 7. Core System Behavior
The system follows this workflow:

1. Capture transaction data from a wallet-connected interface.
2. Parse the transaction into structured fields such as destination, function type, value, token amount, and approval amount.
3. Build a local transaction graph from the parsed entities and run a graph-based predictor together with any supporting baseline checks.
4. Run targeted simulation for transactions that require effect estimation beyond parsing and model output.
5. Generate a short explanation.
6. Display a risk card with severity, reason, and recommendation.

## 8. Risk Card Format
Every alert should use the same simple structure so the output is consistent.

Required format:

• Expected action: what the transaction does.
• Risk reason: what triggered the warning.
• Possible impact: what could happen if the user signs.
• Recommended action: what the user should do next.
• Severity: low, medium, high, or critical.

Example:

• Expected action: This transaction gives the contract permission to spend your USDC.
• Risk reason: The approval amount is unlimited.
• Possible impact: If the contract is malicious or compromised, it could later move all of your approved USDC.
• Recommended action: Reject unless you trust the contract and need unlimited approval.
• Severity: High.

## 9. Technical Components

### Frontend
Purpose: capture transaction context and display explanations.

Main tasks:

• Build a wallet-connected interface.
• Show a compact risk card before signing.
• Keep explanations short and readable.
• Support a consistent severity system.

Tools:

• React.
• A wallet SDK such as wagmi or RainbowKit.

### Backend
Purpose: parse transaction data, run model inference or supporting checks, and generate explanations.

Main tasks:

• Accept transaction payloads from the frontend.
• Normalize transaction fields.
• Build the transaction-centered graph and apply trained predictor inference plus any supporting baseline checks.
• Trigger simulation when needed.
• Return structured risk results.

Tools:

• FastAPI.
• Python.

### Graph ML Layer
Purpose: score transactions with a trained graph model that improves on the baseline heuristics while remaining explainable enough for the risk card.

Main tasks:

• Prepare labeled transaction examples.
• Build a local heterogeneous graph around each transaction.
• Engineer node, edge, and scalar features.
• Train and evaluate candidate graph models.
• Save the chosen model artifact and load it for inference.
• Expose model output to the explanation layer.

Tools:

• PyTorch.
• PyTorch Geometric or a similar graph-learning framework.

### Simulation Layer
Purpose: estimate likely transaction outcomes when simple parsing is not enough.

Main tasks:

• Run selected transaction simulations.
• Extract relevant state changes.
• Feed the results into the explanation layer.

Tools:

• Foundry or another EVM-compatible simulation environment.

### Explanation Layer
Purpose: convert technical findings into user-facing guidance.

Main tasks:

• Map model outputs and supporting analysis signals to short explanations.
• Keep wording consistent across alerts.
• Avoid technical jargon.
• Ensure each message answers what, why, and next step.

## 10. Recommended Graph Model Design

Best recommended approach: use a transaction-centered heterogeneous graph rather than a full chain-wide graph.

Reasoning:

• The current app analyzes one pending transaction at a time.
• The current request schema already contains the seed entities needed to build a local graph.
• A local heterogeneous graph is easier to train, debug, and explain within project scope than a global evolving on-chain graph.

Recommended node types:

• Wallet or initiating address.
• Destination contract.
• Spender or operator address when present.
• Token asset.
• Transaction anchor node.
• Optional simulation-effect or token-flow nodes.

Recommended edge types:

• `initiates` from wallet to transaction.
• `targets` from transaction to destination contract.
• `approves` from wallet to spender.
• `requests_allowance_for` from transaction to token.
• `transfers_value_to` or `transfers_token_to` for visible asset movement.
• `routes_to` for swap or routing hops.
• `grants_operator_to` or `grants_privilege_to` for broad permissions.
• `triggers_effect` from transaction to simulated effect nodes.

Recommended first model:

• Use an R-GCN or similarly relation-aware heterogeneous GNN as the first predictor baseline.
• Pool the learned graph embedding around the transaction anchor node.
• Concatenate that graph embedding with scalar features such as chain ID, method selector, approval amount, token amount, ETH value, and simulation profile.
• Feed the combined representation into an MLP head that predicts risk family probabilities and a severity score.

Why this is the best fit for the current repo:

• It preserves the existing frontend API.
• It uses the current parser as the graph-construction preprocessor.
• It works well with typed relations such as approval, transfer, privilege grant, and token flow.
• It is more practical for a student project than a larger graph transformer unless the dataset grows substantially.

Recommended training target:

• Start with multi-label prediction for approval risk, destination risk, and simulation risk.
• Derive or jointly train a severity output from those class probabilities.
• Keep the current explanation templates, but populate evidence from the most important graph relations and scalar features.

## 11. Workstreams

### Workstream 1: User and Risk Understanding
Goal: decide which user problems matter most before building the prototype.

Tasks:

• Review common pre-signature attack scenarios.
• Identify the most confusing transaction types for users.
• Compare possible warning styles.
• Decide the final list of supported risks.

Output:

• A short document listing supported risks, user needs, and explanation rules.

### Workstream 2: Graph Risk Modeling
Goal: build the smallest useful trained predictor and integrate it into the risk analysis engine.

Tasks:

• Implement transaction parsing.
• Define the heterogeneous graph schema for transaction entities and flows.
• Prepare a labeled dataset for core transaction-risk categories.
• Train and evaluate the first graph-model baseline.
• Integrate model inference into backend analysis.
• Add simulation-based checks for selected transaction types.
• Return structured results with severity and reason.

Output:

• A backend service that receives a transaction and returns model-informed risk findings.

### Workstream 3: Explanation and Interface
Goal: present findings in a way users can act on quickly.

Tasks:

• Design the risk card layout.
• Write explanation templates.
• Connect frontend inputs to backend results.
• Test different wording lengths and styles.

Output:

• A wallet-connected prototype interface.

### Workstream 4: Evaluation
Goal: verify that the system helps users understand risk better.

Tasks:

• Create example transaction scenarios.
• Compare user decisions with and without explanations.
• Check whether users understand the warnings.
• Record which explanation formats work best.

Output:

• An evaluation summary with findings and limitations.

## 12. Minimum Deliverables
The project deliverables are:

• A list of supported risks.
• A working backend for transaction parsing, graph construction, and trained predictor inference.
• A frontend that shows pre-signature risk cards.
• At least three end-to-end demonstration scenarios.
• A graph-based trained predictor model with documented evaluation results.
• A short evaluation based on user scenarios, usability review, or user testing.
• A final report describing what was built, what worked, and what remains future work.

## 13. Success Criteria
The project is successful when it meets these criteria:

• The system correctly identifies the chosen risk patterns.
• The graph model performs credibly on held-out evaluation data and is documented clearly.
• The explanations are short and understandable.
• Users can describe what a flagged transaction would do.
• Users can distinguish more serious warnings from less serious ones.
• The prototype is stable enough to demonstrate full flow from transaction input to risk output.

## 14. Practical Evaluation Metrics
Use evaluation criteria that match the project goal.

Evaluation metrics:

• Model quality: F1, precision and recall, or AUROC on held-out graph-labeled examples.
• Severity calibration: Do model confidence and assigned severity align with expected risk levels?
• Comprehension: Can users explain the flagged transaction in their own words?
• Decision quality: Do users make safer decisions with the warning shown?
• Clarity: Do users find the explanation understandable and not overloaded?
• Actionability: Do users know what to do next after reading the alert?
• Trust calibration: Do users react differently to low-risk and high-risk cases?

## 15. Timeline

### Phase 1: Definition and Planning
• Finalize supported risk categories.
• Define explanation format.
• Select tech stack and architecture.

### Phase 2: Core Backend Development
• Build transaction parser.
• Define the graph schema and prepare training data.
• Implement the first graph-model baseline.
• Add structured output format.

### Phase 3: Frontend and Integration
• Build risk card UI.
• Connect frontend to backend.
• Run end-to-end transaction examples.

### Phase 4: Simulation and Refinement
• Add targeted simulation for transactions that cannot be explained by parsing and model output alone.
• Refine graph features and relation definitions.
• Improve wording and severity mapping.
• Fix gaps found during testing.

### Phase 5: Evaluation and Finalization
• Run scenario-based evaluation or user testing.
• Summarize findings.
• Document limitations and future work.

## 16. Project Risks and Mitigation

### Risk: Scope becomes too large
Mitigation: keep the implementation limited to a few explainable risks.

### Risk: Technical analysis works but explanations are not useful
Mitigation: test explanation wording early and revise it based on user feedback.

### Risk: Simulation adds too much complexity
Mitigation: use simulation only for cases where model output alone is not enough.

### Risk: Graph model complexity exceeds dataset quality
Mitigation: start with a compact local graph and an R-GCN baseline before attempting more complex graph architectures.

### Risk: Users ignore the interface
Mitigation: keep warnings short, visible, and action-oriented.

## 17. Immediate Next Steps
The team will do these first:

1. Finalize the exact supported risk cases.
2. Define the backend response format for risk results.
3. Define the transaction-centered graph schema and feature plan.
4. Build the basic transaction parsing pipeline.
5. Prepare the initial labeled dataset and train the first graph-model baseline.
6. Create the first frontend mockup of the risk card and prepare sample transactions for testing.

## 18. Long-Term Extensions
Only after the core prototype works should the team consider:

• More transaction categories.
• Better contract reputation sources.
• Broader multi-chain support.
• Stronger graph model families beyond the first R-GCN-style baseline.
• Shared or on-chain risk registries.

## 19. Final Positioning
ChainSentry is an explainability-first, user-centered blockchain safety prototype. The main value of the project is not the number of technical modules it includes, but whether it helps users make better pre-signature decisions through clear and actionable warnings.

## 20. Two-Student Task Split
To keep execution efficient, the work is split into one integration-heavy role and one ML-and-evaluation-heavy role.

### Student 1: Technical Lead
Primary responsibility: most of the programming and system integration.

Tasks:

• Build the backend service in FastAPI and Python.
• Implement transaction parsing and structured risk output.
• Implement the baseline checks, simulation hooks, and graph-integration points needed for the predictor.
• Set up the simulation workflow.
• Build the frontend integration with wallet connection and risk card display.
• Connect frontend and backend into an end-to-end working system.
• Handle debugging, testing, and final technical integration.

### Student 2: ML and Product Lead
Primary responsibility: trained predictor development, evaluation, documentation, and user-facing refinement.

Tasks:

• Review pre-signature attack scenarios and finalize supported risks with Student 1.
• Define the graph schema, labels, features, and data requirements for the trained predictor.
• Train and evaluate the graph-based predictor model.
• Integrate the graph predictor into the backend with Student 1's support.
• Define explanation rules, severity labels, and risk card wording.
• Design the risk card layout and user-facing content.
• Prepare example transaction scenarios for testing and demonstration.
• Plan and run the evaluation process, including user scenarios or usability review.
• Document model metrics, limitations, and future work.
• Support frontend content updates, testing, and presentation preparation.

### Shared Responsibilities
• Finalize scope and technical decisions together.
• Review explanation quality together before each demo.
• Test the full system together before submission.
• Prepare the final report and presentation together.