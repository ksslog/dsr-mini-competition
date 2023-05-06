from matplotlib import pyplot as plt
import seaborn as sns

def post_processing(X_train, y_train, pred_train, X_val, y_val, pred_val, png_file_name):

    # Plot results
    fig, ax = plt.subplots()
    sns.lineplot(data = X_train, x=X_train.index, y=y_train, label="train_labels")
    sns.lineplot(data = X_train, x=X_train.index, y=pred_train, label="prediction_on_training_set")
    sns.lineplot(data = X_val, x=X_val.index, y=y_val, label="validation_labels")
    sns.lineplot(data = X_val, x=X_val.index, y=pred_val, label="prediction_on_validation_set")
    ax.legend()
    # ax.set_title(model_name)
    fig.show()
    plt.savefig(png_file_name)

    # Write out csv with prediction

    # df2 = X_test[['iq', 'year', 'weekofyear']]
    # df2['city']=df2.iq.replace(to_replace=1, value="iq")
    # df2.drop('iq', axis=1, inplace=True)

    # df2['total_cases'] = pred.tolist()

    # df3 = df2[['city','year','weekofyear','total_cases']]

    # df3.to_csv(csv_file_name, index=False)
    
    pass
