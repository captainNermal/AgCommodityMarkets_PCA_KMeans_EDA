"""
Author: Joshua Smith 
Title: Global Agricultural Commodity Markets: G7 & BRICS Nations Trade

Data: United Nations Agricultural Association - containing frequently updated trade matricies between nations (import volumes, monetary values, 
      trade partners, commodity types, etc.)
      - Supporting VIX and inflation figures from FRED (Saint Lewis Federal Reserve)

Questions:

1.	What key insights regarding global agricultural trade relationships can be derived through preliminary EDA?

2.	What does global agricultural trade look like between and among G7 and BRICS nations?

3.	How has the USD/ Tonne fluctuated from 2000 to 2022 for the three agricultural commodities with the highest import 
    trade volumes, and is there any relationship between these price changes and the VIX?

4.	Using PCA and KMeans Clustering, what are the key traits that characterize the top three agricultural commodities defined 
    by the highest trade volumes, between 2000 and 2022?

5.	Earth's most valuable crop: How can US agricultural purchasers leverage real opportunity cost methodology to identify arbitrage 
    opportunities in the global trade of corn (maize) by analyzing the largest exporters and importers among G7 and BRICS nations? 

"""

# Library imports - general
import pandas as pd
import numpy as np
from scipy.stats import hmean
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns

# Library imports - specific
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

#read data 
def read_data():
    trade_data_raw = pd.read_csv('FAOSTAT_data_en_6-5-2024.csv', encoding='utf-8')
    vix_overlap = pd.read_csv('VIXCLS.csv',encoding='utf-8')
    inflationUSA_overlap = pd.read_csv('CPIINF.csv', encoding='utf-8')

    return trade_data_raw, vix_overlap, inflationUSA_overlap

#clean dara
def clean_data(trade_data_raw,vix_overlap,inflationUSA_overlap):
    """
    Trade Data
    """
    #preliminary assessment
    trade_cleaned = trade_data_raw.iloc[:,:16] # remove empty columns past required data 
    #print(trade_cleaned.isna().any(axis=0).sum()) # check #rows na's
    #print(trade_cleaned.dtypes) #check inital datatypes

    #adjustments 
    trade_cleaned['Year'] = pd.to_numeric(trade_cleaned['Year'], errors='coerce') # year to datetime for time series analysis 
    
    #final - series to df
    finalTradeData_cleaned = pd.DataFrame(trade_cleaned)

    """
    Vix Data
    """
    vix_overlap['Year'] = vix_overlap['DATE'].str[:4]
    vix_overlap['Year'] = vix_overlap['Year'].astype(int) 
    vix_overlap['VIXCLS'] = pd.to_numeric(vix_overlap['VIXCLS'], errors='coerce')
    vix_overlap = vix_overlap.dropna(subset=['VIXCLS'])

    """
    Inf Data
    """
    inflationUSA_overlap['Year'] = inflationUSA_overlap['DATE'].str[:4]
    inflationUSA_overlap['Year'] = inflationUSA_overlap['Year'].astype(int)
    inflationUSA_overlap['FPCPITOTLZGUSA'] = pd.to_numeric(inflationUSA_overlap['FPCPITOTLZGUSA'], errors='coerce')
    inflationUSA_overlap = inflationUSA_overlap.dropna(subset=['FPCPITOTLZGUSA'])
    inflationUSA_overlap = inflationUSA_overlap.rename(columns = {'FPCPITOTLZGUSA':'INFRATE'})

    return finalTradeData_cleaned, vix_overlap,inflationUSA_overlap

#eda correlation
def eda_correlation(finalTradeData_cleaned):
    #commodities import quantities
    importquan_data_corr = finalTradeData_cleaned[finalTradeData_cleaned['Element'] == 'Import Quantity'][['Reporter Countries', 'Item', 'Value']]
    pivot_data_import = importquan_data_corr.pivot_table(index='Reporter Countries', columns='Item', values='Value')
    correlation_matrix_import_quan = pivot_data_import.corr()
    #generate
    plt.figure(figsize=(16, 12))
    sns.heatmap(correlation_matrix_import_quan, annot=True, cmap='coolwarm', linewidths=.5, fmt='.2f')
    plt.title('Correlation Heatmap of Import Quantities of Commodities')
    #plt.show()

    #commodities import quantities
    export_quan_data_corr = finalTradeData_cleaned[finalTradeData_cleaned['Element'] == 'Export Quantity'][['Reporter Countries', 'Item', 'Value']]
    pivot_data_export = export_quan_data_corr.pivot_table(index='Reporter Countries', columns='Item', values='Value')
    correlation_matrix_export_quan = pivot_data_export.corr()
    #generate
    plt.figure(figsize=(16, 12))
    sns.heatmap(correlation_matrix_export_quan, annot=True, cmap='coolwarm', linewidths=.5, fmt='.2f')
    plt.title('Correlation Heatmap of Export Quantities of Commodities')
    #plt.show()

#eda boxplot
def eda_boxplot(finalTradeData_cleaned):
    #desired columns for all below - note we see lots of "outliers" as we are dealing with import values regardless of commodites for country which some are signifigantly less than others 
    #suggeests wide range of diverse products mostly low value prodcuts in high or low volunme or high prodcuts in low qunatity
    importval_data_box = finalTradeData_cleaned[finalTradeData_cleaned['Element'] == 'Import Value'][['Reporter Countries', 'Value']]
    plt.figure(figsize=(20, 10))
    sns.violinplot(x='Reporter Countries', y='Value', data=importval_data_box)
    plt.yscale('log') # log y scale to adjust for high values 
    plt.xticks(rotation=90)
    plt.title('Violin Plot of Import Values by Country')
    plt.xlabel('Country')
    plt.ylabel('Import Value (log scale) 1000s USD')
    plt.tight_layout()
    #plt.show()

    exportval_data_box = finalTradeData_cleaned[finalTradeData_cleaned['Element'] == 'Export Value'][['Reporter Countries', 'Value']]
    plt.figure(figsize=(20, 10))
    sns.violinplot(x='Reporter Countries', y='Value', data=exportval_data_box)
    plt.yscale('log') # log y scale to adjust for high values 
    plt.xticks(rotation=90)
    plt.title('Violin Plot of Export Values by Country1')
    plt.xlabel('Country')
    plt.ylabel('Export Value (log scale) 1000s USD')
    plt.tight_layout()
    #plt.show()

# barchat import export by country
def barchart_importtoexport_bycountry(finalTradeData_cleaned):
    #data prep
    i_to_e_prepimport = finalTradeData_cleaned[finalTradeData_cleaned['Element'] == 'Import Value'][['Reporter Countries', 'Value','Year']]
    i_to_e_prepexport = finalTradeData_cleaned[finalTradeData_cleaned['Element'] == 'Export Value'][['Reporter Countries', 'Value','Year']]
    sums_imports_val = pd.DataFrame(i_to_e_prepimport[i_to_e_prepimport['Year'].between(2018,2022)].groupby('Reporter Countries')['Value'].agg("sum"))
    sums_exports_val = pd.DataFrame(i_to_e_prepexport[i_to_e_prepexport['Year'].between(2018,2022)].groupby('Reporter Countries')['Value'].agg("sum"))
    #merge
    merged_sums_iande_df = pd.merge(sums_imports_val,sums_exports_val,on='Reporter Countries')
    merged_sums_iande_df = merged_sums_iande_df.rename(columns={'Value_x':'Import Value 1000s USD','Value_y':'Export Value 1000s USD'})
    merged_sums_iande_df.reset_index(inplace=True)
    #print(merged_sums_iande_df)

    #vis
    df_melted = merged_sums_iande_df.melt(id_vars=['Reporter Countries'], 
                                        value_vars=['Import Value 1000s USD', 'Export Value 1000s USD'], 
                                        var_name='Type', value_name='Value')
    #print(df_melted)
    # # Plotting with Seaborn
    plt.figure(figsize=(14, 7))
    sns.barplot(x='Reporter Countries', y='Value', hue='Type', data=df_melted, palette='viridis')
    plt.yscale('log')
    plt.xlabel('Country')
    plt.ylabel('Value (1000s USD)')
    #plt.ticklabel_format(style='plain', axis='y')
    plt.title('Cummulative Import and Export Amounts (1000s USD) of Agricultural Commodities by Country (2000-2022)')
    plt.xticks(rotation=90)
    plt.tight_layout()
    #plt.close('all')
    #plt.show()

#linechart dollar per 1000s USD/Tonne to VIX
def linechart_dollar_perton(finalTradeData_cleaned,vix_overlap):
    """
    Identify top 3 volumetric commodites for import trade volume in Tonnes for all years
    """
    quant_commidity_prepimport = finalTradeData_cleaned[finalTradeData_cleaned['Element'] == 'Import Quantity'][['Item', 'Value']]
    #print(quant_commidity_prepimport)
    cummulative_vol_bycommod = quant_commidity_prepimport.groupby('Item').agg("sum")
    cummulative_vol_bycommod['Value'] = cummulative_vol_bycommod['Value'].astype(int)
    #identify max 3
    top3_quantvol_commodites = cummulative_vol_bycommod.nlargest(3,'Value')
    print(top3_quantvol_commodites) #- appears wheat soy beans and corn
    
    #get dollar per tonne 
    prelim_dollar_pertonne = finalTradeData_cleaned[(finalTradeData_cleaned['Element'].isin(['Import Quantity', 'Import Value'])) & 
                                                    (finalTradeData_cleaned['Item'].isin(['Soya beans', 'Wheat', 'Maize (corn)'])) &
                                                    finalTradeData_cleaned['Year'].between(2000,2024)] 
    
    
    pivoted_data = prelim_dollar_pertonne.pivot_table(index=['Year', 'Item'], 
                                         columns='Element', values='Value', aggfunc='sum').reset_index()
    
    pivoted_data['Dollars per Tonne (1000s USD)'] = pivoted_data['Import Value'] / pivoted_data['Import Quantity']

    pivoted_data = pivoted_data.dropna(subset='Dollars per Tonne (1000s USD)')
    #print(pivoted_data)

    top3_items = top3_quantvol_commodites.index.tolist()
    plt.figure(figsize=(12, 6))
    for item in top3_items:
        item_data = pivoted_data[pivoted_data['Item'] == item]
        plt.plot(item_data['Year'], item_data['Dollars per Tonne (1000s USD)'], marker='o', label=item)
    
    """
    Vix Alteration
    """
    #print(vix_overlap)
    harmonicmean = vix_overlap[vix_overlap['Year'].between(2000,2022)].groupby('Year')['VIXCLS'].agg(hmean)
    #print(harmonicmean)

    fig, ax1 = plt.subplots(figsize=(12, 6))
        # Plot the primary data
    for item in pivoted_data['Item'].unique():
        item_data = pivoted_data[pivoted_data['Item'] == item]
        ax1.plot(item_data['Year'], item_data['Dollars per Tonne (1000s USD)'], label=f'{item} Dollars per Tonne')

    # Customize the plot for the first y-axis
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Dollars per Tonne (1000s USD)')
    ax1.set_title('1000s USD/ Tonne and VIX Harmonic Mean (2000-2022)')
    ax1.grid(True)

    # Create a second y-axis for the VIX harmonic mean
    ax2 = ax1.twinx()
    ax2.plot(harmonicmean.index, harmonicmean.values, color='red', marker='x', label='VIX Harmonic Mean')
    ax2.set_ylabel('VIX Harmonic Mean', color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + [lines_2[0]], labels_1 + [labels_2[0]], loc='upper left')

    #plt.show()

#pca
def pca(finalTradeData_cleaned):
    #top 3 ag soy beans, wheat and corn 
    prep_pca_transform = finalTradeData_cleaned[finalTradeData_cleaned['Item'].isin(['Maize (corn)','Wheat','Soya beans'])][[
        'Year','Item','Value','Element','Reporter Countries']]

    prep_pca_transform = prep_pca_transform.pivot_table(index = ['Year','Item'], columns = 'Element', values='Value').reset_index()
    #print(prep_pca_transform)
    prep_pca_transform = prep_pca_transform.dropna(subset=['Year','Item','Export Quantity','Export Value','Import Quantity','Import Value'])
    # pd.set_option('display.max_rows', None)
    # pd.set_option('display.max_columns', None)
    # pd.set_option('display.width', None)
    # pd.set_option('display.max_colwidth', None)
    #print(prep_pca_transform)

    features_pca = ['Year','Export Quantity', 'Export Value', 'Import Quantity', 'Import Value']
    x_pca = prep_pca_transform.loc[:,features_pca].values
    x_pca = StandardScaler().fit_transform(x_pca)
    y_pca = prep_pca_transform.loc[:,['Item']]

    pca = PCA(n_components=3)
    principal_components = pca.fit_transform(x_pca)
    principal_df = pd.DataFrame(data=principal_components,columns=['PCA1','PCA2','PCA3'])
    final_df = pd.concat([principal_df, prep_pca_transform[['Item']],prep_pca_transform[features_pca].reset_index(drop=True)], axis=1)
    
    print(pca.explained_variance_ratio_)
    print(pca.components_)

    #vis 
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('PCA Analysis')
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('PCA3')

    categories = ['Maize (corn)','Soya beans','Wheat']
    colors = ['r','g','b']

    for cat, col in zip(categories, colors):
        index_to_keep = final_df['Item'] == cat
        ax.scatter(final_df.loc[index_to_keep,'PCA1'],final_df.loc[index_to_keep, 'PCA2'],final_df.loc[index_to_keep, 'PCA3'],c=col,s=50)

    ax.legend(categories)
    #plt.show()

    return final_df

#kmeans
def kmeans(final_df):
    #KMeans
    number_clusters =3 # no overlap 
    kmeans = KMeans(n_clusters=number_clusters)
    kmeans.fit(final_df[['PCA1','PCA2','PCA3']])
    final_df['Cluster'] = kmeans.labels_

    # Plotting in original feature space - reduced diminsinoality data and plotting in orignal diminesion space which is valid
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title('K-Means Clustering Following PCA')
    ax.set_xlabel('PCA1')
    ax.set_ylabel('PCA2')
    ax.set_zlabel('PCA3')

    for cluster in range(number_clusters):
        index_to_keep = final_df['Cluster'] == cluster
        ax.scatter(
            final_df.loc[index_to_keep, 'PCA1'], 
            final_df.loc[index_to_keep, 'PCA2'], 
            final_df.loc[index_to_keep, 'PCA3'], 
            c=colors[cluster % len(colors)], 
            s=50, 
            label=f'Cluster {cluster}'
        )
    ax.legend()
    #plt.show()
    # Kmeans analysis
    #cluster component analysis
    for cluster in range(number_clusters):
        cluster_data = final_df[final_df['Cluster'] == cluster]
        commodity_counts = cluster_data['Item'].value_counts(normalize=True) * 100
        print(f'Cluster {cluster} composition:\n{commodity_counts}\n')
    
    #feauter analysis
    features = ['Export Quantity', 'Export Value', 'Import Quantity', 'Import Value']
    cluster_profiles = final_df.groupby(['Cluster', 'Item'])[features].agg(['mean', 'std']).reset_index()
    cluster_profiles_clean = cluster_profiles.dropna()
    #cluster_profiles_adj = cluster_profiles_clean.drop(['PCA1','PCA2','PCA3'],axis=1)
    print(cluster_profiles_clean)

#opp cost
def opp_cost(finalTradeData_cleaned, inflationUSA_overlap):
    #identify imports from brics and g7 to USA 
    opp_cost_prelim = finalTradeData_cleaned[finalTradeData_cleaned['Item'] == 'Maize (corn)'][['Reporter Countries','Partner Countries','Item','Element','Value']]
    #print(opp_cost_prelim)
    opp_cost_prelim = opp_cost_prelim[(opp_cost_prelim['Reporter Countries'] == 'United States of America') & (opp_cost_prelim['Element'].isin(['Import Quantity','Import Value']))]
    #print(opp_cost_prelim)
    opp_cost_prelim_agg = pd.DataFrame(opp_cost_prelim.groupby(['Partner Countries','Element'])['Value'].agg(sum)).reset_index() # allwo for weighted average
    #print(opp_cost_prelim_agg)

    #average inflation harmonic
    average_inf = inflationUSA_overlap['INFRATE'].mean()

    #pivot 
    opp_cost_pivot = opp_cost_prelim_agg.pivot(index = 'Partner Countries', columns = 'Element', values = 'Value').reset_index()
    opp_cost_pivot = opp_cost_pivot.rename(columns={'Import Quantity':'Cumm. Import Quantity','Import Value':'Cumm. Real Import Value'})
    #adj for inflation
    opp_cost_pivot['Cumm. Real Import Value'] = opp_cost_pivot['Cumm. Real Import Value'] / (1 + (average_inf/100))
    opp_cost_pivot['Cumm. Real Import Value'] = opp_cost_pivot['Cumm. Real Import Value'].astype(int) # get rid of scientif notation
    #real(adj for inf) 
    opp_cost_pivot['WA 1000s USD/Tonne Real - USA Pays'] = opp_cost_pivot['Cumm. Real Import Value'] / opp_cost_pivot['Cumm. Import Quantity'] # same as taking a weighted average 


    #identify benchmark country - brazil
    benchmark_cost = opp_cost_pivot['WA 1000s USD/Tonne Real - USA Pays'].min()
    #calc Opp cost - The Opportunity Cost per Unit column shows the additional cost incurred per unit when importing from that country compared to the cheapest option available.
    opp_cost_pivot['Opportunity Cost Per Tonne To Benchmark'] = opp_cost_pivot['WA 1000s USD/Tonne Real - USA Pays'] - benchmark_cost
    print(opp_cost_pivot)

    #vis
    plt.figure(figsize=(12, 8))
    norm = plt.Normalize(opp_cost_pivot['Opportunity Cost Per Tonne To Benchmark'].min(), opp_cost_pivot['Opportunity Cost Per Tonne To Benchmark'].max())
    colors = cm.RdYlBu_r(norm(opp_cost_pivot['Opportunity Cost Per Tonne To Benchmark']))

    bars = plt.barh(opp_cost_pivot['Partner Countries'], opp_cost_pivot['Opportunity Cost Per Tonne To Benchmark'], color=colors)
    plt.xlabel('Opportunity Cost Per Tonne (1000s USD)')
    plt.ylabel('Partner Countries')
    plt.title('Real Opportunity Cost 1000s USD/Tonne to Benchmark for Each US G7 and BRICS Trading Partners')
    plt.gca().invert_yaxis()  # Invert y-axis to have the highest bar on top

    # Add value labels to the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height() / 2, f'${width:.2f}', 
                ha='left', va='center', color='black', fontsize=10)
    
    plt.tight_layout()
    plt.show()

#main wrapper
def main():
    trade_data_raw, vix_overlap, inflationUSA_overlap = read_data()
    finalTradeData_cleaned,vix_overlap,inflationUSA_overlap = clean_data(trade_data_raw,vix_overlap,inflationUSA_overlap)
    eda_correlation(finalTradeData_cleaned)
    eda_boxplot(finalTradeData_cleaned)
    barchart_importtoexport_bycountry(finalTradeData_cleaned)
    linechart_dollar_perton(finalTradeData_cleaned,vix_overlap)
    final_df = pca(finalTradeData_cleaned)s
    kmeans(final_df)
    opp_cost(finalTradeData_cleaned, inflationUSA_overlap)

#call main
main()



