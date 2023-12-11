import pandas as pd

def update_data(df1,df2, emb_model): # check 

    # Merge dataframe
    merged_df = pd.merge(df1.drop(columns=['emb']), df2, how='outer', indicator=True)

    # check deleted data
    del_list = merged_df.query('_merge == "left_only"').drop(columns=['_merge']).index

    # Remove deleted data
    df1.drop(del_list, axis=0, errors='ignore',inplace=True)

    # Create dataframe of the New and Updated (Changed) data 
    add_df = merged_df.query('_merge == "right_only"').drop(columns=['_merge'])
    
    if len(add_df) == 0: # if there is no new or updated data, return df1
        return df1
    
    # preprocess data
    # add_data = extract(add_df, 'question', 'answer')

    # embedd data
    # corpus_embeddings = embed_corpus(emb_model, add_data)
    # corpus_embeddings = text_to_emb(add_df['answer'].tolist(), model, tokenizer)
    corpus_embeddings = emb_model.embed_documents(add_df['answer'].tolist())

    # convert corpus embedding to list
    add_df['emb'] = corpus_embeddings.tolist()

    # append the dataframe
    df1 = pd.concat([
        df1, add_df
    ]).reset_index(drop=True)

    return df1