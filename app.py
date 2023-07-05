import pandas as pd
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import shap
from flask import Flask, jsonify, request
from module import disease_shap_exp
import json
import os
import pandas as pd
from flask import Flask,render_template
from flask_cors import CORS
import requests
    
app = Flask(__name__,static_url_path="", static_folder='', template_folder='')
CORS(app)

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

@app.route("/get_chart/<pdb_code>")
def get_chart(pdb_code):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go
    from graphein.protein.graphs import construct_graph
    from graphein.protein.config import ProteinGraphConfig
    from graphein.protein.edges.distance import add_hydrogen_bond_interactions, add_ionic_interactions, add_peptide_bonds
    from graphein.protein.visualisation import plot_distance_matrix
    from graphein.protein.visualisation import plot_distance_landscape
    from graphein.protein.visualisation import plotly_protein_structure_graph
    from graphein.protein.visualisation import asteroid_plot
    # pdb_code = request.args[0]
    print(f"----------------{pdb_code}------------------")
    # Create backbone graph
    config = ProteinGraphConfig()
    simple_graph = construct_graph(config, pdb_code=pdb_code)

    # Create backbone graph with additional interactions
    edge_funcs = [add_hydrogen_bond_interactions, add_ionic_interactions, add_peptide_bonds]
    config = ProteinGraphConfig(edge_construction_functions= edge_funcs)
    complex_graph = construct_graph(config, pdb_code=pdb_code)
    
    fig=plot_distance_matrix(simple_graph)
    
    # contact_map = (simple_graph.graph["dist_mat"] > 10).astype(int) # Threshold distance matrix
    # fig.add_trace(plot_distance_matrix(g=simple_graph, dist_mat=contact_map)) # Plot contact map
    fig.write_html("chart1.html", include_plotlyjs=False, full_html=False)

    fig=plot_distance_landscape(simple_graph)
    fig.write_html("chart2.html", include_plotlyjs=False, full_html=False)

    fig=plotly_protein_structure_graph(
        G=complex_graph,
        plot_title="Residue level graph with bonds",
        colour_nodes_by="residue_number",
        colour_edges_by="kind",
        node_size_min=20,
        node_size_multiplier=1
        )
    
    fig.write_html("chart3.html", include_plotlyjs=False, full_html=False)
    
    import matplotlib.pyplot as plt
    from typing import List, Callable
    import networkx as nx
    import graphein.rna as gr

    config = gr.RNAGraphConfig()
    g = gr.construct_graph(pdb_code=pdb_code, config=config)
    fig=gr.plotly_rna_structure_graph(g, colour_nodes_by="residue_name", plot_title="RNA Structure with nodes colored by Residue Name")

    fig.write_html("chart4.html", include_plotlyjs=False, full_html=False)

    fig=gr.plotly_rna_structure_graph(g, colour_nodes_by="chain", plot_title="RNA Structure with nodes colored by Chain")
    fig.write_html("chart5.html", include_plotlyjs=False, full_html=False)

    from functools import partial

    config = gr.RNAGraphConfig(
        edge_construction_functions=[
            partial(gr.add_k_nn_edges, k=5, long_interaction_threshold=-1),
            partial(gr.add_distance_threshold, threshold=3, long_interaction_threshold=-1),
            ])

    #g = gr.construct_graph(pdb_code=pdb_code, config=config)
    fig=gr.plotly_rna_structure_graph(g, colour_nodes_by="residue_name", node_size_min=1, plot_title="RNA Structure with Alternative Edge Construction")
    fig.write_html("chart6.html", include_plotlyjs=False, full_html=False)

    with open('chart.html', 'w') as outfile:
        with open('chart1.html', 'r') as file1:
            outfile.write("<div class='row'><h4>Protein Structures</h4><p>Protein distance matrix is widely used in various protein sequence analyses, and mainly obtained by using pairwise sequence alignment scores or protein sequence homology <a href='https://pubmed.ncbi.nlm.nih.gov/24110625/#:~:text=Protein%20distance%20matrix%20is%20widely,a%20combination%20of%20these%20features'>(from NLM library)</a></p><p><a href='https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3018873/'>Importance of Distance plots</a><div class='col-md-6'><div align='center' class='container'>")
            outfile.write(file1.read())
        with open('chart2.html', 'r') as file2:
            outfile.write("</div></div><div class='col-md-6'><div align='center' class='container'>")
            outfile.write(file2.read())
        with open('chart3.html', 'r') as file3:
            outfile.write("</div></div></div><div class='row'><div class='col-md-12'><div align='center' class='container'>")
            outfile.write(file3.read())
        with open('chart4.html', 'r') as file4:
            outfile.write("</div></div></div><div class='row'><h3>RNA Structures</h3><div class='col-md-6'><h4>By Residue Name</h4><p>Residue name:There are 20 amino acids in total. Each amino acids have amines amino parts and carboxylic parts. Residues are what is left overs but they are actually what make the amino acid unique.</p><div align='center' class='container'>")
            outfile.write(file4.read())
        with open('chart5.html', 'r') as file5:
            outfile.write("</div></div><div class='col-md-6'><h4>By Chain</h4><p>The next level of protein structure, secondary structure, refers to local folded structures that form within a polypeptide due to interactions between atoms of the backbone. (The backbone just refers to the polypeptide chain apart from the R groups so all we mean here is that secondary structure does not involve R group atoms.) The most common types of secondary structures are the A; helix and the B; pleated sheet.</p><div align='center' class='container'>")
            outfile.write(file5.read())
        with open('chart6.html', 'r') as file6:
            outfile.write("</div></div></div><div class='row'><h4>Alternative Edge Construction Schemes</h4><p>Most of the spatial edge constructions functions used in the protein API should apply to RNA. Here we construct a graph using both edges based on <b>K-NN connectivity (using  k=5)</b> and spatial edges, joining two nodes if they are within  4\AA  of one another.</p><div class='col-md-12'><div align='center' class='container'>")
            outfile.write(file6.read())
            outfile.write("</div></div></div>")
    return render_template("chart.html")

@app.route("/charts")
def charts():
    return render_template("charts.html")

@app.route("/")
def index():
    return render_template("./index.html")

@app.route("/knowledgeGraph")
def knowledgeGraph():
    return render_template("./knowledgeGraph.html")

@app.route("/dgvGraph")
def dgvGraph():
    return render_template("./dgvGraph.html")

@app.route("/treePage")
def treePage():
    return render_template("./treePage.html")

@app.route("/prediction-exp", methods=["POST"])
def prediction_exp():
    """
    request format:
    {
        "prompt": "",
    }
    Response:
    {
        "predictions": [
            {
                "code": str,
                "probability": float,
                "disease": str,
                "shap_word_weights": [float],
                "shap_word_list": [str],
                "shap_base_value": float, # idk what this does
            } 
        ],
    }
    """
    data = request.json
    prompt = data.get("prompt")
    shap_values, codes, probs, long_desc, short_desc = disease_shap_exp(prompt)
    predictions = [{"code": code,
                    "probability": prob,
                    "disease": ld,
                    "short_desc": sd,
                    "shap_word_weights": shap_values[:, :, code].values[0],
                    "shap_base_value": shap_values[:, :, code].base_values[0],
                    "shap_word_list": shap_values[:, :, code].data}
                   for code, prob, ld, sd in zip(codes, probs, long_desc, short_desc)]

    print(predictions)
    response = {"predictions": predictions}
    return json.dumps(response, cls=NumpyEncoder)


@app.route("/tree-map", methods=["GET"])
def tree_map():
    json_file = os.path.join("data", "ICD", "icd-9-tree.json")
    data = json.load(open(json_file))
    return data

@app.route("/pdb", methods=["GET"])
def get_pdb():
    df = pd.read_csv('data/pdb.csv', dtype='str')
    pdb_codes = df['pdb'].unique().tolist()
    return jsonify({"pdb_codes": pdb_codes})

@app.route("/knowledge-graph", methods=["POST"])
def knowledge_graph():
    """
    request: {"codes": [str]}
    response: {
        "nodes": [{"id": str, "label": str, "node_type": Union["Drug", "Disease"]}],
        "links": [{"source": str, "target": str}]
    }
    """
    MAX_DRUGS_PER_DIS = 10
    df = pd.read_csv(
        'data/ICDdrug/MEDI_wPrevalence_Published_1.csv', dtype='str')
    df['ICD9'] = df['ICD9'].str.replace('.', '')
    codes = request.json.get("codes")

    temp_df = df.loc[df['ICD9'].isin(codes), :]  # ['ICD9'].unique()

    # --- nodes ---
    # diseases
    tdf1 = temp_df.groupby(['ICD9', 'ICD9_STR'], as_index=False).first()
    tdf1['node_type'] = 'Disease'
    node = tdf1.loc[:, ["ICD9", "ICD9_STR", "node_type"]].rename(
        columns={"ICD9": "id", "ICD9_STR": "label"}).to_dict("records")
    # drugs
    temp_df=temp_df.groupby("ICD9").head(MAX_DRUGS_PER_DIS)
    tdf1 = temp_df.groupby(['RXCUI_IN', 'DRUG_DESC'], as_index=False).first()
    tdf1['node_type'] = 'Drug'
    node.extend(tdf1.loc[:, ["RXCUI_IN", "DRUG_DESC", "node_type"]].rename(
        columns={"RXCUI_IN": "id", "DRUG_DESC": "label"}).to_dict("records"))

    # --- links ---
    # links = temp_df.loc[:, ['ICD9', 'RXCUI_IN']].rename(
    #     columns={'RXCUI_IN': 'target', 'ICD9': 'source'}).to_dict('records')
    links = []
    for code in codes:
        code_df = temp_df.loc[temp_df['ICD9']==code, :]
        if len(code_df['RXCUI_IN'].unique()) > MAX_DRUGS_PER_DIS:
            code_df = temp_df.loc[temp_df['ICD9']==code, :].head(MAX_DRUGS_PER_DIS)
        links.extend(code_df.loc[:, ['ICD9', 'RXCUI_IN']].rename(
        columns={'RXCUI_IN': 'to', 'ICD9': 'from'}).to_dict('records'))
    resp = {"nodes": node, "links": links}
    return jsonify(resp)

@app.route("/DGV", methods=["POST"])
def dgv():
    '''
    request: {"diseases": [{"code":str, "vocabularyName": str}]}
    '''
    diseases_list = request.json.get("diseases")
    codes = request.form.get("codes")
    names = request.form.get("names")
    header = {"Authorization": "Bearer 5c38faf061e63ebcc6a642919d484cf3b13fe2f0"}
    base_url = "https://www.disgenet.org/api"
    vocabulary = "icd9cm"
    #diseases_list = dict(zip(icdcodes['code'], icdcodes['vocabularyName']))
    # diseases_list = [{"code":"001","vocabularyName":"Cholera"},{"code":"001.0","vocabularyName":"Cholera"}]
    node_list=[]
    links=[]
    for disease in diseases_list:
        gda_url = f"/gda/disease/{vocabulary}/{disease.get('code')}"
        # gda_url = f"/enrichment/genes"
        node_list.append({"id":disease.get('code'),"label":disease.get('vocabularyName'),"type":"disease"})
        response = requests.get(base_url + gda_url, headers=header )
        if response.status_code == 200:
            #print(response.json())
            response_dict = response.json()
            for i,resp in enumerate(response_dict):
                if i<5:
                    if resp.get('gene_symbol') not in [node.get('id') for node in node_list]:
                        node_list.append({"id":resp.get('gene_symbol'),"label":resp.get('gene_symbol'),"type":"gene"})
                    #print(resp.get('geneid'),resp.get('gene_symbol'), resp.get('disease_name'),disease)
                    links.append({'from': resp.get('gene_symbol'), 'to': disease.get('code')})

        vda_url = f"/vda/disease/{vocabulary}/{disease.get('code')}"
        # gda_url = f"/enrichment/genes"

        response = requests.get(base_url + vda_url, headers=header )
        if response.status_code == 200:
            #print(response.json())
            response_dict = response.json()
            for i,resp in enumerate(response_dict):
                if i<5:
                    if resp.get("variantid") not in [node.get('id') for node in node_list]:
                        node_list.append({"id":resp.get('variantid'),"label":resp.get('variantid'),"type":"variant"})
                    if resp.get("gene_symbol") not in [node.get('id') for node in node_list]:
                        node_list.append({"id":resp.get('gene_symbol'),"label":resp.get('gene_symbol'),"type":"gene"})
                    # node_list.append({"id":resp.get('variantid'),"label":resp.get('variantid'),"type":"variant"})
                    #print(resp.get('geneid'),resp.get('gene_symbol'), resp.get('disease_name'),disease)
                    links.append({'from': resp.get("gene_symbol"), 'to': resp.get('variantid')})
                    links.append({'from': disease.get('code'), 'to': resp.get('variantid')})
                    #print(i.get('variantid'),i.get('gene_symbol'), i.get('disease_name'),disease)
    resp = {"nodes": node_list, "links": links}
    return jsonify(resp)

@app.route("/dgvdiseases", methods=["POST"])
def dgvdiseases():
    df=pd.read_csv('data/dgv/disease_mappings.tsv',dtype='str',sep='\t')
    df=df.loc[df['vocabulary']=='ICD9CM']
    df=df[pd.to_numeric(df['code'], errors='coerce').notnull()]
    df=df.groupby("code").first().reset_index()
    icdcodes=df[['code','vocabularyName']]
    diseases_list = dict(zip(icdcodes['code'], icdcodes['vocabularyName']))
    return jsonify(diseases_list)
if __name__ == '__main__':
    app.run(debug=True)