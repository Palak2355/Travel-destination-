import nbformat as nbf

nb = nbf.read('travel_destination.ipynb', as_version=4)
for cell in nb.cells:
    if cell.cell_type == 'code':
        source = cell.source
        if 'sns.barplot' in source and 'palette=\'plasma\'' in source:
            cell.source = source.replace('palette=\'plasma\'', 'hue=\'Feature\', palette=\'plasma\', legend=False')
nbf.write(nb, 'travel_destination.ipynb')
