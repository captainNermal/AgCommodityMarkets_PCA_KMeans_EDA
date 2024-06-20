# AgCommodityMarketsTrade_PCA_KMeans_EDA
##### Joshua Smith 
## Project Description & Why This Project &#x1F4D3;

One of the oldest and most enduring mediums of exchange, even predating the Neolithic period (~10,000–3,000 BCE), was physical assets or tangible goods; whether that be livestock, consumables such as wheat or barley, or even salt or other precious refined minerals.

With the global population growing at an average CAGR of 1.89%, the challenge of sustainable food production on a global scale has become more critical than ever. Rising global food prices exacerbate food insecurity and poverty; a situation worsened by geopolitical conflicts like the Russia-Ukraine war and the lingering effects of COVID-19-related labor shortages. These spikes in global food prices or the raw inputs that go into creating consumable food products have not only been seen numerous times throughout history but stress the importance of nations effectively acquiring agricultural commodities (Maize, Wheat, Barley, Refined Sugar, etc.) to effectively feed mass populations.

<p align="center">
  <b> "With the world's population expected to hit 9.7 billion by 2050, agriculture needs to become more productive and sustainable. Technology can help transform the global food production system and mitigate its impact on the climate and environment." - World Economic Forum </b>
</p>

Not only is this topic crucial the longevity of the human race - but commodities (agriculture, soft commodities, or other commodities), more specifically, alternative assets (to that of which commodities fall into) are becoming more and more prominent in financial markets. As equity, fixed income, or other traditional asset classes become oversaturated, alternative asset classes (to that of which commodities fall into) and derivatives pertaining, although often much more volatile than traditional assets have grown immensely popular to hedge or bolster portfolios and returns/ alpha.

<p align="center">
  <b> "Alterantive investment [private equity, real estate, and natural resources or commodities] can generate higher returns than traditional investments but come with higher risk" - David Swensen | managed Yales endowment fund </b>
</p>

With this personal and professional appreciation for commodities, particularly agricultural commodities in mind, I hope to explore the five following research questions throughout this article:

1. What key insights regarding global agricultural trade relationships can be derived through preliminary EDA?

2. What does global agricultural trade look like between and among G7 and BRICS nations?

3. How has the USD/ Tonne fluctuated from 2000 to 2022 for the three agricultural commodities with the highest import trade volumes, and is there any relationship between these price changes and the VIX?

4. Using PCA and KMeans Clustering, what are the key traits that characterize the top three agricultural commodities, defined by the highest trade volumes, between 2000 and 2022?

5. Earth’s most valuable crop: How can US agricultural purchasers leverage opportunity cost methodology to identify arbitrage opportunities in the global trade of corn (maize) by analyzing the largest exporters and importers among G7 and BRICS nations?


## Data &#x1F5C3;

The data that I will be working with is from the Food and Agriculture Association of the United Nations which contains information on all global trade relationships pertaining to agricultural crops, livestock, or other consumable products and is updated very regularly.

The data can be shaped to develop a comprehensive trade matrix and thus contains information on the following variables:

· Domain Code: ‘TM’ — Indicates the domain of the data (Trade Matrix).

· Domain: ‘Detailed trade matrix’ — Describes the type of data.

· Reporter Country Code (M49): Numerical codes representing reporting countries.

· Reporter Countries: Names of reporting countries (e.g., Argentina, Australia).

· Partner Country Code (M49): Numerical codes representing partner countries.

· Partner Countries: Names of partner countries (e.g., Argentina, Australia).

· Element Code: Numerical codes for different elements (e.g., 5610 for Import Quantity).

· Element: Types of trade data (e.g., Import Quantity, Export Value).

· Item Code (CPC): Numerical codes for different items (e.g., 115 for Barley).

· Item: Names of items (e.g., Barley, Maize).

· Year Code: Numerical codes representing the years.

· Year: The specific years of the data.

· Unit: Units of measurement (e.g., t for metric tons, 1000 USD for values).

· Value: The actual data values (e.g., import/export quantities or values).

· Flag: Codes indicating the source or type of data (e.g., ‘X’, ‘A’, ‘I’).

· Flag Description: Descriptions of the flags (e.g., Official figure, Imputed value).

Note that the data was obtained from a first-degree source, and thus was already quite clean with the acceptation of a few null or empty values scattered throughout. As these anomalies were often less than 1/50th of the data, they could be dropped or derived via a mean or linear interpolation without leading to influential outliers, skewed data, or misleading results.

Supporting datasets were also drawn from FRED (the Saint Lewis Federal Reserve) to pull in inflationary and VIX figures (to adjust nominal metrics to real and gauge global market fear and volatility in relation to agricultural commodity volumes and prices).


## Tech Stack: Libraries &#x1F4F6;
* Numpy: used for statistical data manipulation
* Pandas: used for statistical data manipulation
* Matplot: used for visualizations
* Seaborn: used for visualizations
* Scripy: used for mathematic application - harmonic mean
* sklearn: used for PCA and KMeans analysis
