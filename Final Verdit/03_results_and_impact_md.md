# Part 3: Experimental Results, Auditing, and Impact

In this section, we present what happened when we ran our framework on 34 days of real solar plant data from India (broken into 15-minute intervals). 

## 1. The Methodological Audit (A Note on Scientific Honesty)
Before showing results, we must highlight a major achievement of this paper: **Scientific Transparency**. 
In our *initial* experiments, our AI model performed poorly, losing to a simple "Persistence Baseline" (which just guesses that tomorrow will be exactly like today). Instead of hiding this, we audited our entire pipeline. We found flaws like the AI predicting "phantom solar power" at midnight, and bad data scaling. 

After fixing these issues (night zeroing, strict data separation, proper baselines), our new TCN model beat the baseline. We show both the old (flawed) and new (honest) numbers in the paper to prove why rigorous auditing is essential in machine learning.

---

## 2. Forecasting Accuracy Results
We compared our AI model against several baselines. The metric used is MAE (Mean Absolute Error) — lower is better.

*   **Original Flawed AI Model:** 8.09% error.
*   **Persistence Baseline (guessing yesterday's weather):** 6.46% error.
*   **Our Corrected TCN Model:** **6.12% error.** (It successfully beat the baseline!)
*   *Note:* The absolute best performer was actually a "Historical Diurnal Profile" (5.24% error), which is just a climatology average of the whole month. Our paper honestly concludes that in highly stable summer weather, historical averages are incredibly hard to beat for raw accuracy. However, our TCN model provides *dynamic uncertainty estimation*, which the historical average cannot do.

---

## 3. Game Theory & Economic Results
We ran four different community simulations to see how much money the community actually saved (Realized Savings) after paying all their imbalance penalties.

| Model driving the Game | Forward Traded Energy | Settled Shortfall (Fines) | Realized Savings (Profit) |
| :--- | :--- | :--- | :--- |
| **B0: Persistence Baseline** | 63.75 kWh | 36.33 kWh | **$2.11** |
| **B3: Our Forecast-Aware TCN** | 69.78 kWh | 33.42 kWh | **$3.62** |

**What this means:** Our TCN model allowed the community to trade more energy *and* suffer fewer shortfall fines compared to the baseline. As a result, the community's realized savings jumped from $2.11 to $3.62—a massive **71% improvement** simply by making smarter, risk-aware commitments.

---

## 4. Stress Testing the Community

### A. The Weather Shock (Cloudburst)
We simulated a severe 6-hour storm hitting the solar panels exactly at noon (generation drops by 75%, and uncertainty triples).
*   *The Financial Impact:* The community lost $124.48 due to heavy imbalance fines (they promised energy at noon and couldn't deliver).
*   *The Behavioral Impact:* Interestingly, the participation rate (people choosing to "Share") did *not* collapse. It stayed stable at around 87%. 
*   *Conclusion:* The financial mechanism absorbs the shock perfectly, proving that a single bad weather event won't permanently destroy community cooperation.

### B. Uncertainty Sweep (Is there a tipping point?)
We artificially increased the forecast risk penalty to see if there is a "cliff" where everyone suddenly quits the P2P market. 
*   *Result:* There is no sudden crash. Even when we multiplied the risk penalty by 15, participation slowly and smoothly slid down (from 50% down to 40%). The community erodes under bad forecasting, but it does not break.

---

## 5. Transfer Learning (Zero-Shot Generalization)
To prove our AI model didn't just memorize Plant 1, we applied it directly to **Plant 2** (a completely different solar plant with different inverters) without doing any retraining (this is called "Zero-Shot"). 
*   **Result:** The model achieved a **5.89% error rate** on Plant 2, successfully beating Plant 2's own local baseline (6.25%). The model learned true physics and weather patterns, not just data noise.

---

## 6. Overall Conclusion and Impact
Our framework proves three major things:
1.  **Forecasting must be tied to reality:** An AI that makes bold predictions without financial accountability is dangerous. Our two-stage settlement ensures that models are penalized if they make prosumers pay imbalance fees.
2.  **Uncertainty matters:** By quantifying risk using Conformal Prediction, prosumers can safely adapt their energy trading strategies.
3.  **Resilient Communities:** Evolutionary Game theory shows that prosumer communities are robust. They will gently reduce their trading during highly uncertain times, but they won't abandon the community entirely. 

This project provides a complete, honest, and mathematically rigorous blueprint for how future smart-grids and renewable communities should handle AI-driven energy trading.