import networkx as nx
from numpy.random import uniform

G = nx.DiGraph() #or G = nx.MultiDiGraph()
G.add_node('3')
G.add_node('2')
G.add_node('1')
G.add_node('0')
G.add_edge('3', '2')
G.add_edge('2', '3')
G.add_edge('3', '1')

pos = nx.spring_layout(G)

fig, ax  = plt.subplots(1, 1, figsize=(6.4, 4.8))
nx.draw(G, pos, with_labels=True, ax=ax, node_size=600, node_color='lightcoral',connectionstyle='arc3, rad = 0.3', arrowstyle='-')
fig.savefig('Figures/I_Implementation_A.png', dpi=600,
            bbox_inches='tight')
plt.show()



N = 10000
x = [0, 1, 2, 3]
probs = [0, 0, 0, 0]
for i in range(N):
    r =uniform(0, 1)
    if r > 0.5:
        probs[3] += 1/N
    elif r > 0.16666:
        probs[2] += 1/N
    else:
        probs[1] += 1/N
theory = [0, 0, 0.16666, 0.33333, 0.5, 0.5]
x_theory = [-0.5, 0, 1, 2, 3, 3.5]

plt.figure(figsize=(6.4, 4.8))
plt.bar(x, probs, color='black', label='Simulated Data $N=1000$')
plt.plot(x_theory, theory, linestyle='--', drawstyle='steps-mid', color='red', label='Theoretical Distribution')
plt.ylabel("Probabiity of Attachment")
plt.xlabel("Node Number")
plt.legend()
fig.savefig('Figures/I_Implementation_B.png', dpi=600,
            bbox_inches='tight')

