import graphviz

# Create a new directed graph
dot = graphviz.Digraph(comment='Diabetes Prediction ML Pipeline')
dot.attr(rankdir='LR')

# Add nodes for each step in the pipeline
dot.node('A', 'Raw Data\n(CSV File)', shape='folder')
dot.node('B', 'Data Preprocessing\n- Label Encoding\n- Feature Selection', shape='box')
dot.node('C', 'Feature Scaling\n(StandardScaler)', shape='box')
dot.node('D', 'Dimensionality Reduction\n(PCA)', shape='box')
dot.node('E', 'Model Training\n(Random Forest)', shape='box')
dot.node('F', 'Prediction\n(0 or 1)', shape='diamond')

# Add edges to connect the nodes
dot.edge('A', 'B')
dot.edge('B', 'C')
dot.edge('C', 'D')
dot.edge('D', 'E')
dot.edge('E', 'F')

# Save the diagram
dot.render('ml_pipeline', format='png', cleanup=True)