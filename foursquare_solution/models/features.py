cat_features = ['found', 'found_name', 'found_country', 'simple_sim', 'country']
train_cols = [
    'kdist', 'kneighbors', 'found', 'kdist_name',
    'kneighbors_name', 'found_name', 'kdist_country', 'kneighbors_country',
    'found_country', 'simple_sim', 'haversine_dist', 'country',
    'euclidian_dist', 'name_dist',  'name_dist_cos', 'name_sim', 'name_gesh',
    'name_leven', 'name_jaro', 'name_lcs', 'name_len_diff', 'name_nleven',
    'name_nlcsk', 'name_nlcs', 'address_sim', 'address_gesh',
    'address_leven', 'address_jaro', 'address_lcs', 'address_len_diff',
    'address_nleven', 'address_nlcsk', 'address_nlcs', 'city_gesh',
    'city_leven', 'city_jaro', 'city_lcs', 'city_len_diff', 'city_nleven',
    'city_nlcsk', 'city_nlcs', 'state_sim', 'state_gesh', 'state_leven',
    'state_jaro', 'state_lcs', 'state_len_diff', 'state_nleven',
    'state_nlcsk', 'state_nlcs', 'zip_gesh', 'zip_leven', 'zip_jaro',
    'zip_lcs', 'url_sim', 'url_gesh', 'url_leven', 'url_jaro', 'url_lcs',
    'url_len_diff', 'url_nleven', 'url_nlcsk', 'url_nlcs', 'phone_gesh',
    'phone_leven', 'phone_jaro', 'phone_lcs', 'categories_sim',
    'categories_gesh', 'categories_leven', 'categories_jaro',
    'categories_lcs', 'categories_len_diff', 'categories_nleven',
    'categories_nlcsk', 'categories_nlcs', 'country_sim', 'country_gesh',
    'country_leven', 'country_jaro', 'country_lcs', 'country_len_diff',
    'country_nleven', 'country_nlcsk', 'country_nlcs'
]