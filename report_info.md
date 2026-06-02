**FAIMDL RL Project Report Structure**
Maximum 5 pages excluding references

**Template:** https://www.overleaf.com/latex/templates/cvpr-2022-author-kit/qbmjsdxryffn
**Structure:**

**0. Abstract**

A single-paragraph summary of the report. It should briefly state the problem: learning RL control policies in simulation and studying sim-to-real transfer in a sim-to-sim setting. Mention the implemented algorithms and strategies, and summarize the key quantitative results and conclusions.

**1. Introduction**

Project Context: Define the core task of reinforcement learning for robotic control and the sim-to-real transfer problem.

Literature Foundations: Explicitly cite and discuss the provided baseline references on RL, RL in robotics, sim-to-real transfer, PPO/SAC, and domain randomization. Extra references used during development can also be included.

**2. Methodology**

Since the environments and project tasks are common across teams, this section must focus **on the team’s specific implementation choices**, extra optional implementations and pipelines.

Just as reference, students should report:
- Implementation of the different algorithms, including relevant loss functions and design choices.
- Explain which advanced RL argument has been used, the selection of hyperparameters, ecc.
- Domain Randomization: Detail the implemented UDR and ADR strategies, including the chosen mass ranges, adaptation rules, and relevant hyperparameters.
- Anything else that is related to the specific project ideas and implementations

Evaluation Metrics: Clearly define the metrics and protocol used to assess performance for each evaluated configuration. 
Note: you can choose whether to divide this section and the next one by part 1 and part 2 (following project specific parts) or to keep them in the same text without specific subsections. Both options are valid.


**3. Experimental Results**

The Results Table: Present the mandatory results table clearly, including all required training→test configurations.

Quantitative Breakdown: Describe the numbers objectively. Point out where the policies performed well, where they failed, and how UDR/ADR affected the transfer from source to target.

Tables and figures can (and should) be inserted to make the results clear to the reader. 
Example 1: reinforce, reinforce + baseline and actor critic could be compared using a plot with mean and std over time to show the specific features/behaviour of the algorithms.
Example 2: during part 2, the various sim2sim settings are easy to report in a table (with mean and std if tried over different seeds).

**4. Discussion**

This is the most critical analytical section of the report.

**Result Analysis**: Go beyond the numbers and explain why the results happened.

**Methodological Critique**: Reflect on the limitations of the approaches, including , e.g., algorithm stability, hyperparameter sensitivity, UDR/ADR limitations, and the simplified sim-to-sim setting.

**5. Conclusion**

A very short final paragraph summarizing the main outcome: best-performing algorithm, effectiveness of domain randomization, and main limitation or possible future improvement.

**6. References**

A cleanly formatted list of all literature cited throughout the text, adhering strictly to the report template style.

**Formatting Reminder for Students**: The report should be self-contained, which means that if some paragraph is referring to something, it should be introduced earlier.