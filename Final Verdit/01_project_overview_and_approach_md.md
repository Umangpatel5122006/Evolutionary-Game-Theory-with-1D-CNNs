# Part 1: Project Overview, Core Concepts, and Approach

## 1. Introduction: The Problem We Are Solving
In a Renewable Energy Community, households with solar panels (prosumers) can sell their extra energy to their neighbors instead of sending it back to the main grid. This is called **Peer-to-Peer (P2P) trading**. P2P trading is great because neighbors can buy energy cheaper than the grid price, and sellers can sell it for more than the standard grid feed-in rate. Everyone wins.

**The Catch:** To trade energy tomorrow, you have to promise it today based on a **forecast**. But solar generation depends on the weather. If a cloud rolls in, you might not generate the energy you promised. Most current research assumes forecasts are 100% true and ignores what happens when prosumers fail to deliver. Furthermore, if prosumers keep losing money because of bad forecasts, they might stop participating in the community altogether.

**Our Solution:** We created a framework that combines **Machine Learning Forecasting**, **Uncertainty Quantification** (measuring how unsure the forecast is), and **Evolutionary Game Theory** (modeling how human behavior changes over time). Crucially, we enforce a **Two-Stage Settlement**: prosumers make decisions based on forecasts, but they pay actual financial penalties (imbalance tariffs) if they fail to deliver the energy they promised in reality.

---

## 2. Simple Definitions of Important Terms
To make this project easy to understand, here are the key concepts:

*   **Prosumer:** A household that both *produces* (via solar panels) and *consumes* electricity.
*   **P2P (Peer-to-Peer) Trading:** A local market where neighbors buy and sell extra electricity directly with each other.
*   **Share vs. Hoard:** The two choices a prosumer has in our game. 
    *   *Share:* Participate in the P2P market with neighbors.
    *   *Hoard:* Ignore the neighbors and only trade with the main utility grid.
*   **TCN (Temporal Convolutional Network):** A type of AI/Deep Learning model used to analyze time-series data. We use it to predict future solar power generation based on past weather and power data.
*   **Conformal Prediction:** A mathematical way to guarantee how accurate a prediction interval is. Instead of just guessing "it might be between 10kW and 20kW," it provides a mathematical guarantee like "we are 90% sure the actual power will fall in this range."
*   **Evolutionary Game Theory:** A branch of math that studies how populations change their strategies over time. Instead of assuming humans are perfectly rational calculators, it assumes people look at their neighbors and copy whatever strategy is making the most money.
*   **Replicator Dynamics:** The specific mathematical rule in Evolutionary Game Theory that says: *If a strategy is highly profitable, more people will adopt it. If it loses money, people will abandon it.*
*   **Ex-ante vs. Ex-post:** *Ex-ante* means "before the event" (making a commitment based on a forecast). *Ex-post* means "after the event" (paying the actual bill based on what the solar panels actually produced).

---

## 3. The Overall Approach
Our project follows a strict timeline for every 15-minute interval of the day. The architecture chains these steps together:

1.  **Forecast (The Guess):** Our deep learning model (TCN) looks at the past 48 hours of weather and solar data and predicts the next 24 hours of solar generation.
2.  **Uncertainty (The Risk):** The model doesn't just give one number. It gives a range (using Conformal Prediction) so we know how risky the forecast is.
3.  **The Game (The Decision):** Prosumers look at the forecast, their own home's energy demand, and the risk of being wrong. They calculate expected payoffs and choose whether to *Share* or *Hoard*.
4.  **The Reality Check (Two-Stage Settlement):** After the 15-minute interval passes, we look at the *actual* solar data. If a prosumer promised 5 kWh to a neighbor but a cloud blocked the sun and they only generated 2 kWh, they must buy the missing 3 kWh from the main grid at an expensive **imbalance tariff** to fulfill their promise. 

## 4. Why This Matters (The Impact)
By forcing the AI model to live in a "real-world" scenario where mistakes cost money, we proved that highly complex AI models can sometimes be actively harmful if they promise energy that doesn't exist. Our approach protects the community. It ensures that prosumers only make commitments they can keep, reducing expensive grid penalties and keeping the community profitable and cooperative in the long run.