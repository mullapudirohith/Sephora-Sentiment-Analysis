#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().run_line_magic('pip', "install -r '../requirements.txt'")


# In[2]:


import shutil
import os
import zipfile


dependency_path = '../dependencies/'
file_name = 'archive.zip'
file_id = '1PWh1rMqWh0JjkygP-CZbYn-VtL2TMrmB'
files_to_check = ['reviews_0-250.csv', 'reviews_250-500.csv', 'reviews_500-750.csv', 'reviews_750-1250.csv', 'reviews_1250-end.csv']


def download_file(id_of_file, destination):
    os.system('gdown ' + id_of_file + ' -O ' + destination)

def check_files():
    if not os.path.exists(dependency_path):
        return False
    for file_to_check in files_to_check:
        if not os.path.exists(os.path.join(dependency_path, file_to_check)):
            return False
    return True

if not check_files():
    if os.path.exists(dependency_path):
        shutil.rmtree(dependency_path)
    os.makedirs(dependency_path) 
    download_file(file_id, dependency_path + file_name)
    try:
        with zipfile.ZipFile(dependency_path + file_name, 'r') as zip_ref:
            zip_ref.extractall(dependency_path)
        print("Extraction successful.")
        os.remove(dependency_path + file_name)
    except zipfile.BadZipFile:
        print("Error while extracting zip")
    if os.path.exists(dependency_path + '__MACOSX'):
        shutil.rmtree(dependency_path + '__MACOSX')
    for file in files_to_check:
        shutil.move(dependency_path + 'archive/' + file, dependency_path + file)
    shutil.rmtree(dependency_path + 'archive')
else:
    print("Files already exist, skipping download")
    


# In[3]:


import requests
from bs4 import BeautifulSoup

if not os.path.exists('../data/brand_links.txt'):
    brand_lst_link = 'https://sephora.com/brands-list'
    response = requests.get(url=brand_lst_link, headers={'User-Agent': 'Your Custom User-Agent'})
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Scraping brand links and saving them into a list
    brand_link_lst = []
    main_box = soup.find_all(attrs={"data-at": "brand_link", "data-comp": "StyledComponent BaseComponent "})
    
    for brand in main_box:
        brand_link = brand.get('href')
        if not str(brand_link).startswith('/brand'):
            brand_link = '/brand' + brand_link
        brand_link_lst.append("https://www.sephora.com" + brand_link + "/all?pageSize=300")
    
    if not os.path.exists('../data/'):
        os.mkdir('../data/')
    with open('../data/brand_links.txt', 'w') as f:
        for link in brand_link_lst:
            f.write(link + '\n')
            
    print("Saving brand links to /data/brand_links.txt")
    
else:
    print("Extracted brand links found at /data/brand_links.txt, proceeding with locally saved file. To re-extract the brand links, delete the file and run the cell again")


# In[4]:


import json
import pandas as pd


brand_links = []
with open('../data/brand_links.txt', 'r') as f:
    for line in f:
        brand_links.append(line.strip())

if not os.path.exists('../data/product_links.csv'):
    print("Brand links have been loaded from the text file: brand_links.txt")
    
    brand_names = []
    product_names = []
    product_ids = []
    product_links = []
    
    for brand in brand_links:
        brand_name = brand.split("/")[4]
        print("Scraping data for " + brand_name)
        response = requests.get(url=brand, headers={'User-Agent': 'Your Custom User-Agent'})
        soup = BeautifulSoup(response.content, 'html.parser')
        try:
            products = soup.find(attrs={"data-comp": "PageJSON "})
            products = json.loads(products.get_text('script')).get('page').get('nthBrand').get('products')
            for product in products:
                mini_url = product.get('targetUrl')
                product_name = product.get('displayName')
                product_id = product.get('productId')
                product_link = "https://www.sephora.com" + mini_url + "/all?pageSize=300"
    
                brand_names.append(brand_name)
                product_names.append(product_name)
                product_ids.append(product_id)
                product_links.append(product_link)
        except TypeError as e:
            print("No products found for the Brand: " + brand_name)
    
    products_df = pd.DataFrame({
        'brand_name': brand_names,
        'product_name': product_names,
        'product_id': product_ids,
        'product_link': product_links
    })
    products_df.to_csv("../data/product_links.csv")
    print("Saving product links to /data/product_links.csv")
    
else:
    products_df = pd.read_csv('../data/product_links.csv')
    brand_names = products_df['brand_name'].tolist()
    product_names = products_df['product_name'].tolist()
    product_ids = products_df['product_id'].tolist()
    product_links = products_df['product_link'].tolist()
    print("Extracted product links found at /data/product_links.csv, proceeding with locally saved file. To re-extract the product links, delete the file and run the cell again")


# In[5]:


reviews = []
for file in files_to_check:
    reviews.append(pd.read_csv(dependency_path + file, low_memory=False))
reviews_df = pd.concat(reviews, ignore_index=True)


# In[6]:


reviews_df.head()


# In[7]:


reviews_df.describe()


# In[8]:


product_id_list = reviews_df['product_id'].unique()


# In[9]:


len(product_id_list)


# In[10]:


common_products = []
for product in product_ids:
    for product_idx in product_id_list:
        if product == product_idx:
            common_products.append(product)
print("There are " +str(len(common_products)) + " common products")


# In[11]:


reviews_df = reviews_df[reviews_df['product_id'].isin(common_products)]


# In[12]:


len(reviews_df['product_id'].unique())


# In[13]:


products_df = products_df[products_df['product_id'].isin(common_products)]


# In[14]:


products_df.count()


# In[15]:


import time
import json
import re
from io import StringIO


def remove_text_in_parentheses(input_string):
    result = re.sub(r'\([^)]*\)', '', input_string)
    return result.strip()

def get_chemicals_from_link(product_link):
    response = requests.get(url=product_link, headers={'User-Agent': 'User-Agent'})
    soup = BeautifulSoup(response.content, 'html.parser')
    product = soup.find(attrs={"data-comp": "PageJSON "})
    try:
        ingredients_text = json.loads(product.get_text('script')).get('page').get('product').get('currentSku').get(
            'ingredientDesc')
    except Exception as e:
        return []
    if not ingredients_text:
        return []
    ingredients_soup = BeautifulSoup(ingredients_text, 'html.parser')
    names = ingredients_soup.get_text().split(',')
    individual_names = [name for name in names if name.strip()]

    cleaned_names = []
    for name in individual_names:
        if name.startswith('*'):
            continue
        if len(name) < 4:
            continue
        cleaned_name = remove_text_in_parentheses(name)
        if cleaned_name and not cleaned_name.endswith('*'):
            cleaned_names.append(cleaned_name)
            
    ingredients = list(set(cleaned_names))
    chemicals = []

    for ingredient in ingredients:
        retries = 2  # Number of retries
        while retries > 0:
            response = requests.get("https://webbook.nist.gov/cgi/cbook.cgi?Name=" + ingredient + "&Units=SI")
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, 'html.parser')
                title = soup.find('title').get_text()
                if title == "Search Results":
                    best_match = soup.find('main').find('ol').find('li').find('a').get_text()
                    chemicals.append(best_match)
                elif title == "Name Not Found":
                    break
                elif title == "No Matching Species Found":
                    break
                else:
                    chemicals.append(title)
                break 
            else:
                time.sleep(5)
                retries -= 1
        else:
            print(f"Error: Unable to fetch data for ingredient {ingredient}")
    
    return chemicals


if not os.path.exists('../data/products.csv'):
    print("Product links has been loaded from the file: product_links.csv")
    chemicals_lists = []
    total_products = len(products_df)
    
    for i, (index, row) in enumerate(products_df.iterrows(), start=1):
        product_link = row['product_link']
        product_name = row['product_name']
        chemicals = get_chemicals_from_link(product_link)
        chemicals_lists.append(chemicals)
        if chemicals:
            print(f"Chemicals extracted for {i}/{total_products} products: {product_name}")   
            print(chemicals)
        else:
            print("No chemicals found for " + product_name)
    
    products_df['chemicals_list'] = chemicals_lists
    products_df.to_csv("../data/products.csv")
    print("Saving chemicals in products to /data/products.csv")
else:
    products_df = pd.read_csv("../data/products.csv")
    print("Extracted chemicals in products found at /data/products.csv, proceeding with locally saved file. To re-extract the product links, delete the file and run the cell again")


# In[16]:


products_df.info()


# In[17]:


products_df = products_df[["brand_name", "product_name", "product_id", "product_link", "chemicals_list"]]


# In[18]:


products_df.count()


# In[19]:


products_df = products_df[products_df['chemicals_list'].apply(lambda x: len(eval(x)) > 0)]


# In[20]:


products_df.count()


# In[21]:


from collections import Counter
import ast


common_chemicals_list = []

for chemicals_list in products_df['chemicals_list']:
    common_chemicals_list.extend(ast.literal_eval(chemicals_list))

chemicals_frequency = dict(Counter(common_chemicals_list).most_common())

print("Total unique chemicals:", len(chemicals_frequency.keys()))

high_frequency_chemicals_list = []
for chemical in chemicals_frequency.keys():
    if chemicals_frequency[chemical] > 15:
        high_frequency_chemicals_list.append(chemical)


# In[22]:


print("Chemicals that occur in more than 15 products: ")
for high_frequency_chemical in high_frequency_chemicals_list:
    print(high_frequency_chemical)
print("Total High Frequency chemicals: ", len(high_frequency_chemicals_list))


# In[23]:


from ast import literal_eval


products_df['chemicals_list'] = products_df['chemicals_list'].apply(literal_eval)

chemical_counts = pd.Series([chemical for sublist in products_df['chemicals_list'] for chemical in sublist]).value_counts()

frequent_chemicals = chemical_counts[chemical_counts > 15].index.tolist()
filtered_chemicals_list = products_df['chemicals_list'].apply(lambda x: [chemical for chemical in x if chemical in frequent_chemicals]).tolist()

products_df['filtered_chemicals'] = filtered_chemicals_list

print(filtered_chemicals_list)

filtered_products_df = products_df[["brand_name", "product_name", "product_id", "filtered_chemicals"]]
filtered_products_df.to_csv('../data/filtered_products.csv', index=False)


# In[24]:


unique_chemicals = set(chem for sublist in filtered_products_df['filtered_chemicals'] for chem in sublist)

for chemical in unique_chemicals:
    filtered_products_df[chemical] = 0

for index, row in filtered_products_df.iterrows():
    chemicals = row['filtered_chemicals']
    for chem in chemicals:
        filtered_products_df.at[index, chem] = 1


# In[27]:


filtered_products_df.head()


# In[28]:


encoded_df = filtered_products_df.drop(columns=["brand_name", "product_name", "filtered_chemicals"], axis=1)


# In[29]:


encoded_df.head()


# In[ ]:




