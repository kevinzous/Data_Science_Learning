#Replacing codes by indicator names
Dic,Dic_inv={},{}
for i in range(len(county_facts_dictionary)) :
    Dic[county_facts_dictionary.values[i][0]]=county_facts_dictionary.values[i][1]
    Dic_inv[county_facts_dictionary.values[i][1]]=county_facts_dictionary.values[i][0]
train.rename(columns=Dic,inplace=True)
test.rename(columns=Dic,inplace=True)


Dic_inv_sub={k:Dic_inv[k] for k in Cols if k in Dic_inv}
