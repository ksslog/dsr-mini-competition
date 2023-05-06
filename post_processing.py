from matplotlib import pyplot as plt
import seaborn as sns

def post_processing(X_train, y_train, trained, X_test,
                y_test,pred, png_file_name, csv_file_name, model_name):

    # Plot results
    fig, ax = plt.subplots()
    sns.scatterplot(data = X_train, x=X_train.index, y=y_train, label="train")
    sns.scatterplot(data = X_test, x=X_test.index, y=y_test, label="test")
    sns.scatterplot(data = X_test, x=X_test.index, y=pred, label="predict")
    sns.scatterplot(data = X_train, x=X_train.index, y=trained, label="trained")
    ax.legend()
    ax.set_title(model_name)
    fig.show()
    plt.savefig(png_file_name)

    # Write out csv with prediction

    df2 = X_test[['iq', 'year', 'weekofyear']]
    df2['city']=df2.iq.replace(to_replace=1, value="iq")
    df2.drop('iq', axis=1, inplace=True)

    df2['total_cases'] = pred.tolist()

    df3 = df2[['city','year','weekofyear','total_cases']]

    df3.to_csv(csv_file_name, index=False)
    
    pass
