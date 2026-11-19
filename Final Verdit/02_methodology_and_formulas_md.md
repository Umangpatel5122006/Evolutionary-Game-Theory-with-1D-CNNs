# Part 2: Methodology & Mathematical Framework

This section breaks down the technical math and AI models used in our framework. We use a 4-step methodology.

---

## Step 1: The Forecasting Model (Diurnal-Residual TCN)
We built a highly efficient AI model called a **Diurnal-Residual Temporal Convolutional Network**. Solar generation is highly cyclical (the sun rises and sets at the same time). Therefore, instead of making the AI learn everything from scratch, we force it to base its prediction on yesterday's output, and only predict the *difference* (the residual) caused by today's weather.

**The Base Formula:**
$$y_{base} = 0.60 P_{t-96} + 0.40 P_{t-192}$$
*Explanation:* The base prediction is a mix of the solar power from exactly one day ago ($P_{t-96}$, as there are 96 fifteen-minute intervals in a day) and two days ago ($P_{t-192}$). 

**The Final Prediction:**
$$y = y_{base} + 0.5\Delta$$
*Explanation:* The AI model only calculates $\Delta$ (the change due to current clouds, temperature, etc.). We also force the model to output exactly $0$ at night (astronomical night zeroing) to prevent physical impossibilities.

---

## Step 2: Uncertainty Quantification
To calculate risk, we use **Monte Carlo (MC) Dropout** combined with **Split Conformal Calibration**. 
By running the data through the AI model multiple times ($K=50$), we get an average prediction ($\mu_t$) and a standard deviation/spread ($\sigma_t$).

$$\mu_t = \frac{1}{K} \sum_{k=1}^{K} P_t^{(k)}$$
$$\sigma_t = \sqrt{ \frac{1}{K-1} \sum_{k=1}^{K} (P_t^{(k)} - \mu_t)^2 }$$

*Explanation:* $\mu_t$ is our best guess for solar power. $\sigma_t$ tells us how volatile or uncertain that guess is. We apply conformal math to turn this into a strict 90% confidence interval.

---

## Step 3: The Evolutionary Game
Now the 100 prosumers play the game. They can either **Hoard** (trade with grid) or **Share** (trade in P2P market).

### A. P2P Market Pricing
The price of energy inside the community depends on Supply and Demand. 
$$SDR_t = \frac{S_t^+}{S_t^-}$$
*Explanation:* $SDR_t$ is the Supply-Demand Ratio. It divides the total surplus energy ($S_t^+$) by the total energy deficit ($S_t^-$) of the community.

The price is then calculated dynamically:
$$p_{P2P}(t) = p_{fit} + ( p_{grid} - p_{fit} ) \left( 1 - \frac{SDR_t}{1 + SDR_t} \right)$$
*Explanation:* If supply is high, the price drops close to the cheap feed-in tariff ($p_{fit}$). If supply is low, the price rises near the expensive utility grid price ($p_{grid}$).

### B. Risk Penalty and Payoffs
Because forecasting is risky, we subtract a risk penalty $R(t)$ from the expected payoff of sharing. If the AI model is highly uncertain ($\sigma_t$ is large), the penalty is large.

$$R(t) = \left( \frac{s \sigma_t}{C_c} \right)^2$$
*(where $C_c$ is the community capacity and $s$ is a scaling factor).*

The final payoffs for the strategies are:
$$U_S(t) = \pi_S(t) - \lambda_S R(t)$$
$$U_H(t) = \pi_H(t) - \lambda_H R(t)$$
*Explanation:* $U_S$ is the utility (profit) of Sharing, minus the risk penalty. $U_H$ is the utility of Hoarding (which has no risk penalty, because the grid absorbs all risk).

### C. Replicator Dynamics (How behavior changes)
We model human adaptation using this differential equation:
$$\frac{dx}{dt} = x(t) (1 - x(t)) ( U_S(t) - U_H(t) )$$
*Explanation:* $x(t)$ is the percentage of people sharing. If Sharing makes more money than Hoarding ($U_S > U_H$), the rate of sharing goes up. People copy the winning strategy. We calculate this step-by-step using an Euler integration step.

---

## Step 4: Two-Stage Settlement (Honest Accounting)
This is the most critical part of our methodology. Prosumers commit a trade volume ($q_t$) based on the forecast. Later, the sun actually shines, and we measure the real deliverable energy ($q_t^{real}$). 

Any missing energy is the **Shortfall** ($\phi_t$):
$$\phi_t = \max( 0 , q_t - q_t^{real} )$$

The community is punished for this shortfall by having to buy it from the grid at an expensive **imbalance tariff** ($p_{imb}$):
$$C_t^{imb} = \phi_t ( p_{imb} - p_{grid} )$$
*Explanation:* If you lie or are wrong about your solar generation, you pay a heavy fine. This forces the community to value highly accurate, cautious forecasting over wildly optimistic AI predictions.