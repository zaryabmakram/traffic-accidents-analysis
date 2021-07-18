import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

# loading dataset 
df = pd.read_csv("dataset/210619monatszahlenjuni2021monatszahlen2106verkehrsunfaelle.csv")

# dropping last 4 columns
df = df.iloc[:, :-4]

# renaming columns from German to English
# for easy access
df.columns = [
    "Category",
    "Accident_type",
    "Year", 
    "Month",
    "Value"
]

# extracting data other than 2021 
df = df.loc[df['Year'] != 2021]


def visualize_category_yearly(dataset, save_plot=True):
    # extracting yearly data with total accident value
    df_year = dataset.loc[dataset['Month'] == 'Summe']
    
    # grouping data by Year and Category
    df_data = df_year.groupby(['Year', 'Category']).sum()
    
    # creating a bar plot 
    ax = df_data.unstack().plot(
        kind='bar', 
        figsize=(20, 7), 
        log=True,   # log scale
        rot=0, 
        title="Total Number of Accidents per Category over the Years"
    )
    
    # modifying legend  
    ax.legend(
        ["Alkoholunfälle", "Fluchtunfälle", "Verkehrsunfälle"],
        bbox_to_anchor=(1.0, 1.0), # outside, top-right corner
    )
    
    # adding ylabel text 
    ax.set_ylabel("Number of Accidents (log scale)")
    
    # display plot
    plt.show()
    
    # saving plot as .png
    if save_plot:
        fig = ax.get_figure()
        fig.savefig("assets/yearly_plot.png")
    

if __name__ == "__main__":
    visualize_category_yearly(dataset=df)