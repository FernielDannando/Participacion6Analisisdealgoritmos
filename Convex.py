import tkinter as tk
from tkinter import filedialog, messagebox
import scanpy as sc
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from sklearn.cluster import DBSCAN

class App:
    def __init__(self, root):
        """
        Inicializa la aplicación GUI con botones y área de texto.
        """
        self.root = root
        self.root.title("Análisis de Datos UMAP")
        
        # Inicializa los atributos antes de crear los widgets
        self.dbscan_eps = tk.DoubleVar(value=1.1)  # Valor inicial para eps
        self.dbscan_min_samples = tk.DoubleVar(value=5)  # Valor inicial para min_samples
        
        self.create_widgets()
        self.df = None

    def create_widgets(self):
        """
        Crea los widgets de la interfaz gráfica.
        """
        self.load_button = tk.Button(self.root, text="Cargar archivo .h5ad", command=self.load_file)
        self.load_button.pack(pady=10)

        self.text_area = tk.Text(self.root, wrap=tk.WORD, height=20, width=80)
        self.text_area.pack(pady=10)
        
        self.plot_button = tk.Button(self.root, text="Mostrar gráfico UMAP", command=self.plot_data)
        self.plot_button.pack(pady=10)

        # Etiqueta y campo para el parámetro eps
        self.eps_label = tk.Label(self.root, text="Espacio entre puntos:")
        self.eps_label.pack(pady=5)
        
        self.eps_description = tk.Label(self.root, text="Es el parámetro que define la distancia máxima entre puntos para que sean considerados parte del mismo clúster. Ejemplo: 0.5 - 1.5")
        self.eps_description.pack(pady=5)

        self.eps_entry = tk.Entry(self.root, textvariable=self.dbscan_eps)
        self.eps_entry.pack(pady=5)
        
        # Etiqueta y campo para el parámetro min_samples
        self.min_samples_label = tk.Label(self.root, text="Número mínimo de muestras para Agrupar:")
        self.min_samples_label.pack(pady=5)

        self.min_samples_description = tk.Label(self.root, text="Número mínimo de puntos requeridos para formar un clúster. Ejemplo: 5 - 50")
        self.min_samples_description.pack(pady=5)

        self.min_samples_entry = tk.Entry(self.root, textvariable=self.dbscan_min_samples)
        self.min_samples_entry.pack(pady=5)

    def load_file(self):
        """
        Carga un archivo .h5ad y extrae las coordenadas UMAP y los identificadores de clústeres.
        """
        file_path = filedialog.askopenfilename(filetypes=[("Archivos h5ad", "*.h5ad")])
        if not file_path:
            return
        
        try:
            adata = sc.read(file_path)
            umap_coords = adata.obsm.get('X_UMAP', None)
            if umap_coords is None:
                raise ValueError("El archivo .h5ad no contiene datos de UMAP.")
            
            cluster_ids = adata.obs.get('cluster_id', pd.Series([None] * umap_coords.shape[0]))
            
            self.df = pd.DataFrame(umap_coords, columns=['UMAP1', 'UMAP2'])
            self.df['Cluster'] = cluster_ids
            
            self.validate_and_display_data()
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo cargar el archivo: {e}")

    def validate_and_display_data(self):
        """
        Valida los datos cargados y actualiza el área de texto con la información.
        """
        self.text_area.delete(1.0, tk.END)
        if self.df is None or self.df.empty:
            self.text_area.insert(tk.END, "No se cargaron datos válidos.\n")
            return
        
        if self.df['Cluster'].isna().all():
            self.text_area.insert(tk.END, "Todos los valores en 'Cluster' son NaN. Asignando valores predeterminados.\n")
            self.df['Cluster'] = range(len(self.df))
        
        self.df = self.df.dropna(subset=['Cluster'])
        self.df['Cluster'] = self.df['Cluster'].astype(int)
        
        self.text_area.insert(tk.END, "Primeras filas del DataFrame:\n")
        self.text_area.insert(tk.END, str(self.df.head()) + '\n')

        self.text_area.insert(tk.END, "\nDatos de células:\n")
        for index, row in self.df.iterrows():
            coord = (row['UMAP1'], row['UMAP2'])
            cluster = row['Cluster']
            self.text_area.insert(tk.END, f'Celula {index}: Coordenadas={coord}, Clúster={cluster}\n')

    def plot_data(self):
        """
        Muestra un gráfico UMAP con clústeres usando DBSCAN y dibuja las envolturas convexas.
        """
        if self.df is None or self.df.empty:
            messagebox.showerror("Error", "No se ha cargado ningún archivo o los datos están vacíos.")
            return
        
        try:
            eps = self.dbscan_eps.get()
            min_samples = self.dbscan_min_samples.get()
            if eps <= 0 or min_samples <= 0:
                raise ValueError("El valor de eps y min_samples deben ser positivos.")
                
            self.df['DBSCAN_Cluster'] = self.perform_dbscan_clustering(eps, min_samples)
            self.create_umap_plot()
        except ValueError as ve:
            messagebox.showerror("Error de Validación", str(ve))
        except Exception as e:
            messagebox.showerror("Error", f"Error al generar el gráfico: {e}")

    def perform_dbscan_clustering(self, eps, min_samples):
        """
        Realiza el clustering DBSCAN en los datos UMAP.

        Args:
            eps (float): Epsilon (eps) para DBSCAN.
            min_samples (int): Número mínimo de muestras para DBSCAN.

        Returns:
            array: Etiquetas de clúster para cada punto.
        """
        clustering = DBSCAN(eps=eps, min_samples=int(min_samples)).fit(self.df[['UMAP1', 'UMAP2']])
        return clustering.labels_

    def create_umap_plot(self):
        """
        Crea un gráfico UMAP y dibuja las envolturas convexas.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(self.df['UMAP1'], self.df['UMAP2'], c=self.df['DBSCAN_Cluster'], cmap='tab10', alpha=0.6)
        plt.xlabel('UMAP1')
        plt.ylabel('UMAP2')
        plt.title('Visualización de UMAP con Clústeres DBSCAN')
        plt.colorbar(scatter, label='Cluster ID')

        self.plot_convex_hulls(ax)
        plt.show()

    def plot_convex_hulls(self, ax):
        """
        Dibuja el Convex Hull para cada clúster en el gráfico con líneas suaves que coinciden con el color de los clústeres.

        Args:
            ax (matplotlib.axes.Axes): El objeto de ejes en el que se dibujará el Convex Hull.
        """
        grouped = self.df.groupby('DBSCAN_Cluster')
        cmap = plt.get_cmap('tab10')  # Usa el mismo colormap que para el scatter plot
        for cluster_id, group in grouped:
            points = group[['UMAP1', 'UMAP2']].values
            if len(points) >= 3:
                hull = ConvexHull(points)
                color = cmap(cluster_id % cmap.N)  # Usa el índice del cluster para obtener el color
                for simplex in hull.simplices:
                    ax.plot(points[simplex, 0], points[simplex, 1], color=color, alpha=0.5, linewidth=2)
                ax.fill(points[hull.vertices, 0], points[hull.vertices, 1], color=color, alpha=0.2, label=f'Cluster {cluster_id}')

        ax.legend()
        ax.grid(True)

if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
