import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib as mpl
import io
import base64
from PIL import Image
import matplotlib.ticker as ticker
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from scipy.signal import savgol_filter
import time

# Configuration de la page Streamlit
st.set_page_config(
    page_title="Chute verticale avec frottement",
    layout="wide"
)

# Fonction pour créer un bouton de téléchargement de figure
def get_image_download_link(fig, filename="chute_verticale.png", text="Télécharger la figure"):
    buf = io.BytesIO()
    fig.write_image(buf, format="png", width=1200, height=800, scale=2)
    buf.seek(0)
    b64 = base64.b64encode(buf.getvalue()).decode()
    href = f'<a href="data:image/png;base64,{b64}" download="{filename}">{text}</a>'
    return href

st.title("Évolution de la vitesse lors d'une chute verticale avec frottement")
st.markdown("""
Cette application simule la chute d'un objet dans un fluide en prenant en compte :
- La force de gravité
- La poussée d'Archimède
- La force de frottement (proportionnelle à v^n)

L'équation différentielle résolue est : $\\frac{dv}{dt} = g\\left(1-\\frac{\\rho_fV}{m}\\right)-\\frac{k}{m}v^n$
""")

# Création de deux colonnes
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Paramètres")
    
    # Paramètres du corps
    st.markdown("### Caractéristiques du corps")
    shape = st.selectbox("Forme du corps", ["Sphère", "Cube"])
    
    rho_c = st.number_input("Masse volumique du corps (kg/m³)", 
                            min_value=1.0, max_value=20000.0, value=7800.0, step=100.0,
                            help="Ex: Acier=7800, Aluminium=2700, Plomb=11300")
    
    size = st.number_input("Taille (m)", 
                          min_value=0.001, max_value=1.0, value=0.01, step=0.001,
                          help="Rayon pour une sphère, côté pour un cube")
    
    # Paramètres du fluide
    st.markdown("### Caractéristiques du fluide")
    fluid = st.selectbox("Type de fluide", ["Air", "Eau", "Huile", "Personnalisé"])
    
    if fluid == "Air":
        rho_f = 1.2
    elif fluid == "Eau":
        rho_f = 1000.0
    elif fluid == "Huile":
        rho_f = 920.0
    else:
        rho_f = st.number_input("Masse volumique du fluide (kg/m³)", 
                                min_value=0.1, max_value=2000.0, value=1.2, step=0.1)
    
    # Paramètres de frottement avec exposant variable
    st.markdown("### Caractéristiques du frottement")
    friction_type = st.radio("Sélection du type de frottement",
                            ["Prédéfini", "Personnalisé"],
                            index=0,
                            horizontal=True)
    
    if friction_type == "Prédéfini":
        n_option = st.radio("Type de frottement", 
                         [1, 2], 
                         index=1, 
                         format_func=lambda x: "Visqueux (proportionnel à v)" if x == 1 else "Quadratique (proportionnel à v²)",
                         horizontal=True)
        n = float(n_option)
    else:
        n = st.slider("Exposant n dans v^n", 
                      min_value=0.1, max_value=3.0, value=2.0, step=0.1,
                      help="Force de frottement proportionnelle à v^n")
    
    k_default = 0.5 if n >= 1.5 else 0.1
    k = st.number_input(f"Coefficient de frottement k (unités SI)", 
                        min_value=0.001, max_value=10.0, value=k_default, step=0.01)
    
    # Paramètres de simulation
    st.markdown("### Paramètres de simulation")
    g = st.number_input("Accélération gravitationnelle (m/s²)", 
                        min_value=0.0, max_value=20.0, value=9.81, step=0.01)
    
    t_max = st.slider("Durée de simulation (s)", 
                      min_value=1.0, max_value=30.0, value=8.0, step=0.1)
    
    v0 = st.number_input("Vitesse initiale (m/s)", 
                        min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
    
    # Options d'affichage
    st.markdown("### Options d'affichage")
    show_no_drag = st.checkbox("Afficher la courbe sans frottement", value=True)
    show_limit = st.checkbox("Afficher la vitesse limite", value=True)
    show_tangent = st.checkbox("Afficher la tangente à l'origine", value=True)
    show_regimes = st.checkbox("Afficher les régimes (initial/permanent)", value=True)
    show_tau = st.checkbox("Afficher le temps caractéristique τ", value=True)
    
    # Type de graphique
    st.markdown("### Type de graphique")
    graph_type = st.radio("Choisir le type de graphique",
                         ["Interactif avec zoom", "Standard"],
                         index=0,
                         horizontal=True,
                         help="Interactif: Zoom avec la souris, déplacement, etc. Standard: Image statique")
    
    # Bouton pour lancer la simulation
    simulate = st.button("Lancer la simulation", type="primary")
    
    # Statut de calcul
    if 'calculate' not in st.session_state:
        st.session_state.calculate = False

# Calcul du volume et de la surface selon la forme
if shape == "Sphère":
    V = (4/3) * np.pi * size**3  # Volume d'une sphère
    surface = 4 * np.pi * size**2  # Surface d'une sphère
    shape_name = "sphérique"
else:  # Cube
    V = size**3  # Volume d'un cube
    surface = 6 * size**2  # Surface d'un cube
    shape_name = "cubique"

# Calcul de la masse
m = rho_c * V

# Fonction pour calculer la vitesse limite théorique
def calculate_v_lim(n_val, k_val, m_val, g_val, rho_f_val, V_val):
    effective_g = g_val * (1 - rho_f_val * V_val / m_val)
    if effective_g <= 0:
        return 0  # Le corps flotte
    
    if n_val == 1:
        return effective_g * m_val / k_val
    else:
        return (effective_g * m_val / k_val) ** (1/n_val)

# Calculer la vitesse limite
v_lim = calculate_v_lim(n, k, m, g, rho_f, V)

# Calculer le temps caractéristique τ (tau)
# Pour n=1, tau = m/k
# Pour n≠1, nous approximons en utilisant la définition : temps pour atteindre 63% de v_lim
tau = m/k if n == 1 else None  # Calcul exact uniquement pour n=1

# Afficher les informations calculées
with col1:
    st.markdown("### Caractéristiques calculées")
    
    buoyancy_percentage = (rho_f * V * g / (m * g) * 100)
    floating = buoyancy_percentage >= 100
    
    info_text = f"""
    - Volume: {V:.2e} m³
    - Masse: {m:.4f} kg
    - Surface: {surface:.2e} m²
    - Poussée d'Archimède: {rho_f * V * g:.4f} N ({buoyancy_percentage:.2f}% du poids)
    """
    
    if floating:
        info_text += "- ATTENTION: Le corps flotte (poussée > poids)!"
    else:
        info_text += f"- Vitesse limite théorique: {v_lim:.2f} m/s"
        if n == 1:
            info_text += f"\n- Temps caractéristique τ: {tau:.4f} s"
    
    st.info(info_text)

# Définition de l'équation différentielle
def dvdt(t, v):
    # Si l'objet flotte, la vitesse limite est zéro (équilibre)
    if rho_f * V >= m:
        return -k/m * np.abs(v)**(n-1) * v  # Seulement la décélération due au frottement
    return g * (1 - rho_f * V / m) - (k / m) * np.abs(v)**(n-1) * v

with col2:
    if simulate or st.session_state.calculate:
        st.session_state.calculate = True
        
        if floating:
            st.warning("⚠️ Le corps flotte! La simulation montrera l'objet ralentissant jusqu'à l'arrêt.")
        
        # Afficher une barre de progression pour les calculs longs
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Étape 1: Résolution de l'équation différentielle
        status_text.text("Résolution de l'équation différentielle...")
        progress_bar.progress(10)
        
        t_span = (0, t_max)
        solution = solve_ivp(
            dvdt, 
            t_span, 
            [v0], 
            method='RK45',
            dense_output=True,
            rtol=1e-6,
            atol=1e-9
        )
        
        # Étape 2: Création d'une grille de temps plus fine pour le tracé
        status_text.text("Traitement des données...")
        progress_bar.progress(30)
        
        t_fine = np.linspace(0, t_max, 1000)
        v_fine = solution.sol(t_fine)[0]
        
        # Étape 3: Chute sans frottement pour comparaison
        status_text.text("Calcul de la comparaison sans frottement...")
        progress_bar.progress(50)
        
        def dvdt_no_drag(t, v):
            if rho_f * V >= m:
                return 0  # Pas d'accélération si l'objet flotte
            return g * (1 - rho_f * V / m)
        
        solution_no_drag = solve_ivp(
            dvdt_no_drag, 
            t_span, 
            [v0], 
            method='RK45',
            dense_output=True
        )
        
        v_no_drag = solution_no_drag.sol(t_fine)[0]
        
        # Si tau n'est pas déjà calculé (cas n≠1), le calculer numériquement
        if tau is None and not floating and v_lim > 0:
            # Trouver quand la vitesse atteint 63% de v_lim
            target_v = 0.63 * v_lim
            for i, v in enumerate(v_fine):
                if v >= target_v:
                    tau = t_fine[i]
                    break
            else:
                tau = t_max  # Si jamais atteint dans la simulation
        
        # Étape 4: Préparation des données pour le graphique
        status_text.text("Préparation du graphique...")
        progress_bar.progress(70)
        
        # Créer un DataFrame pour Plotly
        df = pd.DataFrame({
            'Temps': t_fine,
            'Vitesse': v_fine,
            'Vitesse sans frottement': v_no_drag if show_no_drag else np.zeros_like(t_fine)
        })
        
        # Étape 5: Création du graphique
        status_text.text("Création du graphique interactif...")
        progress_bar.progress(90)
        fig = go.Figure()
        # Options selon le type de graphique choisi
        if graph_type == "Interactif avec zoom":
            # Créer un graphique Plotly interactif
            fig = go.Figure()
            
            # Ajouter la courbe principale
            fig.add_trace(go.Scatter(
                x=t_fine,
                y=v_fine,
                mode='lines',
                name='Avec frottement',
                line=dict(color='rgb(31, 119, 180)', width=5)
            ))
            
            # Ajouter la courbe sans frottement si demandé
            if show_no_drag and not floating:
                fig.add_trace(go.Scatter(
                    x=t_fine,
                    y=v_no_drag,
                    mode='lines',
                    name='Sans frottement',
                    line=dict(color='rgb(255, 127, 14)', width=2, dash='dash')
                ))
            
            # Ajouter la vitesse limite si demandée
            if show_limit and not floating:
                fig.add_trace(go.Scatter(
                    x=[0, t_max],
                    y=[v_lim, v_lim],
                    mode='lines',
                    name=f'Vitesse limite ({v_lim:.2f} m/s)',
                    line=dict(color='rgb(44, 160, 44)', width=2, dash='dashdot')
                ))
            
            # Ajouter la tangente à l'origine si demandée
            if show_tangent and not floating and v0 == 0:
                # Calcul de la pente à l'origine
                slope = g * (1 - rho_f * V / m)  # dvdt à t=0, v=0
                # Points pour tracer la tangente
                x_tangent = np.array([0, min(tau if tau else t_max/5, t_max/2)])
                y_tangent = slope * x_tangent
                
                fig.add_trace(go.Scatter(
                    x=x_tangent,
                    y=y_tangent,
                    mode='lines',
                    name='Tangente à l\'origine',
                    line=dict(color='blue', width=1.5)
                ))
                
                # Annotation pour la pente
                fig.add_annotation(
                    x=x_tangent[1]/2 + 0.5,
                    y=y_tangent[1]/2 + 0.5,
                    text=f'Pente = {slope:.2f} m/s²',
                    showarrow=True,
                    arrowhead=1,
                    ax=20,
                    ay=-30
                )
            
            # Afficher les temps caractéristiques τ et 5τ
        if show_tau and tau and not floating:
                # Ligne verticale à τ
                fig.add_trace(go.Scatter(
                    x=[tau, tau],
                    y=[0, v_fine[np.abs(t_fine - tau).argmin()]],
                    mode='lines',
                    name=f'τ = {tau:.2f} s',
                    line=dict(color='purple', width=1.5, dash='dot')
                ))
                
                # Annotation pour τ
                fig.add_annotation(
                    x=tau,
                    y=-0.05*v_lim,
                    text='τ',
                    showarrow=False,
                    yshift=10
                )
                
                # Ligne à 5τ si dans la plage de temps
                if 5*tau <= t_max:
                    fig.add_trace(go.Scatter(
                        x=[5*tau, 5*tau],
                        y=[0, v_fine[np.abs(t_fine - 5*tau).argmin()]],
                        mode='lines',
                        name='5τ',
                        line=dict(color='purple', width=1, dash='dot'),
                        opacity=0.7
                    ))
                    
                    # Annotation pour 5τ
                    fig.add_annotation(
                        x=5*tau,
                        y=-0.05*v_lim,
                        text='5τ',
                        showarrow=False,
                        yshift=10
                    )
                
                # Pour n=1, v(τ) = 0.63*v_lim théoriquement
                v_at_tau = np.interp(tau, t_fine, v_fine)
                if n == 1:
                    fig.add_annotation(
                        x=tau + 0.5,
                        y=v_at_tau,
                        text=f'v(τ) ≈ {v_at_tau:.2f} m/s ≈ 63% de v_lim',
                        showarrow=True,
                        arrowhead=1,
                        ax=20,
                        ay=0
                    )
            
         # Par ce bloc corrigé :
        if show_regimes and not floating and tau:
            # Trouver la séparation entre régimes (approximativement à 5τ)
            separation = min(5*tau, t_max)
            
            # Ajouter des zones colorées pour les régimes
            fig.add_vrect(
                x0=0, x1=separation,
                fillcolor="red", opacity=0.1,
                layer="below", line_width=0
            )
            
            # Ajouter l'annotation séparément pour le régime initial
            fig.add_annotation(
                x=separation/2,
                y=1.05,
                yref="paper",
                text="Régime initial",
                showarrow=False,
                font=dict(color="red")
            )
            
        if separation < t_max:
            fig.add_vrect(
                x0=separation, x1=t_max,
                fillcolor="green", opacity=0.1,
                layer="below", line_width=0
            )
            
            # Ajouter l'annotation séparément pour le régime permanent
            fig.add_annotation(
                x=(separation + t_max)/2,
                y=1.05,
                yref="paper",
                text="Régime permanent",
                showarrow=False,
                font=dict(color="green")
            )
    
            # Configurer l'aspect du graphique
            drag_type = "∝v" if n == 1 else f"∝v^{n:.1f}"
            if floating:
                title = f"Simulation d'un corps flottant (densité < {rho_f:.1f} kg/m³)"
            else:
                title = f"Évolution de la vitesse d'un objet {shape_name} de {rho_c:.0f} kg/m³ dans {fluid.lower()} ({rho_f:.1f} kg/m³)<br>avec frottement {drag_type}"
            
            fig.update_layout(
                title=title,
                xaxis_title="Temps (s)",
                yaxis_title="Vitesse (m/s)",
                legend=dict(
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01,
                    bgcolor='rgba(255, 255, 255, 0.8)'
                ),
                plot_bgcolor='rgba(240, 240, 240, 0.8)',
                hovermode="closest",
                height=600
            )
            
            # Ajouter des infobulles détaillées
            fig.update_traces(
                hoverinfo="x+y+name",
                hovertemplate="<b>%{fullData.name}</b><br>Temps: %{x:.3f} s<br>Vitesse: %{y:.3f} m/s<extra></extra>"
            )
            
            # Ajouter un encadré avec les informations importantes
            info_text = f"Coefficient k = {k:.4f}, Exposant n = {n:.2f}<br>"
            info_text += f"Masse = {m:.4f} kg, Volume = {V:.2e} m³<br>"
            
            if floating:
                info_text += f"Corps flottant (poussée = {buoyancy_percentage:.1f}% du poids)"
            else:
                info_text += f"Poussée d'Archimède = {buoyancy_percentage:.1f}% du poids"
            
            fig.add_annotation(
                xref="paper", yref="paper",
                x=0.02, y=0.02,
                text=info_text,
                showarrow=False,
                font=dict(size=10),
                bgcolor="white",
                opacity=0.8,
                bordercolor="black",
                borderwidth=1,
                borderpad=4,
                align="left"
            )
            
            # Configurer les interactions Plotly
            fig.update_layout(
                xaxis=dict(
                    rangeslider=dict(visible=True),
                    type="linear"
                )
            )
            
           
            # Configurer les boutons de navigation et d'export
            fig.update_layout(
                updatemenus=[
                    dict(
                        type="buttons",
                        direction="right",
                        buttons=[
                            dict(
                                args=[{"yaxis.range": [0, max(v_lim * 1.1, max(v_fine) * 1.1) if not floating else max(v_fine) * 1.2]}],
                                label="Réinitialiser Y",
                                method="relayout"
                            ),
                            dict(
                                args=[{"xaxis.range": [0, t_max]}],
                                label="Réinitialiser X",
                                method="relayout"
                            ),
                            dict(
                                args=[{"xaxis.range": [0, min(3*tau if tau else t_max/3, t_max)]}],
                                label="Zoom régime initial",
                                method="relayout"
                            )
                        ],
                        pad={"r": 10, "t": 10},
                        showactive=False,
                        x=0.40,
                        xanchor="left",
                        y=1.0,
                        yanchor="top"
                    )
                ]
            )
            
            # Terminer la barre de progression
            progress_bar.progress(100)
            status_text.text("Graphique interactif prêt ! Utilisez la souris pour zoomer et explorer.")
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()
            
            # Afficher le graphique interactif
            st.plotly_chart(fig, use_container_width=True)
            
            # Bouton pour télécharger la figure
            st.markdown(get_image_download_link(fig), unsafe_allow_html=True)
        
        else:
            # Utiliser matplotlib pour le graphique standard
            plt.style.use('seaborn-v0_8-darkgrid')
            mpl.rcParams['font.family'] = 'DejaVu Sans'
            mpl.rcParams['font.size'] = 12
            mpl.rcParams['figure.figsize'] = (12, 8)
            
            fig, ax = plt.subplots()
            
            # Tracé de la courbe principale
            ax.plot(t_fine, v_fine, '-', color='#1f77b4', linewidth=3, label='Avec frottement')
            
            # Tracé optionnel de la courbe sans frottement
            if show_no_drag and not floating:
                ax.plot(t_fine, v_no_drag, '--', color='#ff7f0e', linewidth=2, label='Sans frottement')
            
            # Tracé optionnel de la vitesse limite
            if show_limit and not floating:
                ax.axhline(y=v_lim, color='#2ca02c', linestyle='-.', linewidth=2, 
                          label=f'Vitesse limite ({v_lim:.2f} m/s)')
            
            # Configuration du graphique
            ax.set_xlabel('Temps (s)', fontsize=14)
            ax.set_ylabel('Vitesse (m/s)', fontsize=14)
            
            # Titre avec informations sur les paramètres
            if floating:
                title = f"Simulation d'un corps flottant (densité < {rho_f:.1f} kg/m³)"
            else:
                drag_type = "∝v" if n == 1 else f"∝v^{n:.1f}"
                title = f"Évolution de la vitesse d'un objet {shape_name} de {rho_c:.0f} kg/m³\n"
                title += f"dans {fluid.lower()} ({rho_f:.1f} kg/m³) avec frottement {drag_type}"
            
            ax.set_title(title, fontsize=16, pad=20)
            
            # Grille et légende
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(loc='best', frameon=True, framealpha=0.9, fontsize=12)
            
            # Ajustement final et affichage
            plt.tight_layout()
            
            # Terminer la barre de progression
            progress_bar.progress(100)
            status_text.text("Graphique standard prêt !")
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()
            
            st.pyplot(fig)
        
        # Onglets pour analyses supplémentaires
        if not floating:
            tab1, tab2, tab3 = st.tabs(["Analyse numérique", "Solutions analytiques", "Données brutes"])
            
            with tab1:
                st.markdown("### Analyse numérique des résultats")
                
                # Calcul du temps pour atteindre différents pourcentages de la vitesse limite
                percentages = [50, 63, 90, 95, 99]
                times_to_reach = []
                
                for percent in percentages:
                    target_v = v_lim * percent / 100
                    for i, v in enumerate(v_fine):
                        if v >= target_v:
                            times_to_reach.append((percent, t_fine[i]))
                            break
                    else:
                        times_to_reach.append((percent, None))
                
                # Affichage des temps pour atteindre les pourcentages
                st.markdown("#### Temps pour atteindre un pourcentage de la vitesse limite")
                time_data = {}
                for percent, t in times_to_reach:
                    if t is not None:
                        time_data[f"{percent}%"] = f"{t:.3f} s{' (τ)' if percent == 63 and n == 1 else ''}"
                    else:
                        time_data[f"{percent}%"] = "Non atteint dans la simulation"
                
                time_df = {"Pourcentage de la vitesse limite": time_data}
                st.write(time_df)
                
                # Affichage des résultats finaux
                st.markdown(f"#### Résultats après {t_max:.1f} secondes")
                final_results = {
                    "Valeur finale": {
                        "Vitesse avec frottement": f"{v_fine[-1]:.3f} m/s",
                        "Pourcentage de la vitesse limite": f"{v_fine[-1]/v_lim*100:.2f}%"
                    }
                }
                
                if show_no_drag:
                    final_results["Valeur finale"]["Vitesse sans frottement"] = f"{v_no_drag[-1]:.3f} m/s"
                    final_results["Valeur finale"]["Différence"] = f"{v_no_drag[-1] - v_fine[-1]:.3f} m/s"
                
                st.write(final_results)
            
            with tab2:
                # Affichage de la solution analytique pour n=1 (cas visqueux)
                if n == 1:
                    st.markdown("#### Solution analytique (pour n=1, frottement visqueux)")
                    st.latex(r"v(t) = v_{lim} \cdot (1 - e^{-t/\tau})")
                    st.latex(r"\tau = \frac{m}{k} = " + f"{m/k:.4f} \text{{ s}}")
                    st.latex(r"v_{lim} = \frac{mg'}{k} = " + f"{v_lim:.4f} \text{{ m/s}} \quad \text{{avec}} \quad g' = g\left(1-\frac{{\rho_f V}}{{m}}\right) = {g * (1 - rho_f * V / m):.4f} \text{{ m/s}}^2")
                    
                    # Utiliser Plotly pour la comparaison
                    # Solution analytique
                    t_anal = np.linspace(0, t_max, 500)
                    v_anal = v_lim * (1 - np.exp(-t_anal / tau))
                    
                    # Créer la figure Plotly
                    fig_compare = go.Figure()
                    
                    # Ajouter la solution numérique
                    fig_compare.add_trace(go.Scatter(
                        x=t_fine,
                        y=v_fine,
                        mode='markers',
                        marker=dict(
                            size=5,
                            color='blue',
                            opacity=0.6
                        ),
                        name='Solution numérique'
                    ))
                    
                    # Ajouter la solution analytique
                    fig_compare.add_trace(go.Scatter(
                        x=t_anal,
                        y=v_anal,
                        mode='lines',
                        line=dict(
                            color='red',
                            width=2
                        ),
                        name='Solution analytique: v<sub>lim</sub>(1-e<sup>-t/τ</sup>)'
                    ))
                    
                    # Configurer le graphique
                    fig_compare.update_layout(
                        title="Comparaison solution analytique vs numérique",
                        xaxis_title="Temps (s)",
                        yaxis_title="Vitesse (m/s)",
                        height=500,
                        hovermode="closest",
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01
                        )
                    )
                    
                    # Calculer l'erreur relative
                    v_anal_interp = np.interp(t_fine, t_anal, v_anal)
                    rel_error = np.abs((v_fine - v_anal_interp) / v_anal_interp) * 100
                    
                    # Afficher l'erreur maximale
                    max_error = np.max(rel_error[~np.isnan(rel_error)])
                    
                    st.plotly_chart(fig_compare, use_container_width=True)
                    st.info(f"Erreur relative maximale: {max_error:.6f}%")
                    
                elif n == 2:
                    st.markdown("#### Solution approchée (pour n=2, frottement quadratique)")
                    st.latex(r"v(t) \approx v_{lim} \cdot \tanh\left(\frac{t \cdot g'}{v_{lim}}\right) \quad \text{avec} \quad g' = g\left(1-\frac{\rho_f V}{m}\right)")
                    
                    # Tracé de la solution approchée vs numérique
                    # Solution approchée
                    t_anal = np.linspace(0, t_max, 500)
                    g_eff = g * (1 - rho_f * V / m)
                    v_anal = v_lim * np.tanh(t_anal * g_eff / v_lim)
                    
                    # Créer la figure Plotly
                    fig_compare = go.Figure()
                    
                    # Ajouter la solution numérique
                    fig_compare.add_trace(go.Scatter(
                        x=t_fine,
                        y=v_fine,
                        mode='markers',
                        marker=dict(
                            size=5,
                            color='blue',
                            opacity=0.6
                        ),
                        name='Solution numérique'
                    ))
                    
                    # Ajouter la solution analytique
                    fig_compare.add_trace(go.Scatter(
                        x=t_anal,
                        y=v_anal,
                        mode='lines',
                        line=dict(
                            color='red',
                            width=2
                        ),
                        name='Solution approchée: v<sub>lim</sub>tanh(tg\'/v<sub>lim</sub>)'
                    ))
                    
                    # Configurer le graphique
                    fig_compare.update_layout(
                        title="Comparaison solution approchée vs numérique",
                        xaxis_title="Temps (s)",
                        yaxis_title="Vitesse (m/s)",
                        height=500,
                        hovermode="closest",
                        legend=dict(
                            yanchor="top",
                            y=0.99,
                            xanchor="left",
                            x=0.01
                        )
                    )
                    
                    # Calculer l'erreur relative
                    v_anal_interp = np.interp(t_fine, t_anal, v_anal)
                    rel_error = np.abs((v_fine - v_anal_interp) / v_anal_interp) * 100
                    
                    # Afficher l'erreur maximale
                    max_error = np.max(rel_error[~np.isnan(rel_error)])
                    
                    st.plotly_chart(fig_compare, use_container_width=True)
                    st.info(f"Erreur relative maximale: {max_error:.6f}%")
                    
                else:
                    st.markdown("#### Solution numérique")
                    st.markdown("Pour n ≠ 1 et n ≠ 2, il n'existe généralement pas de solution analytique simple.")
                    st.markdown("La courbe affichée est obtenue par intégration numérique de l'équation différentielle.")
                    
                    # Équation différentielle
                    st.latex(r"\frac{dv}{dt} = g\left(1-\frac{\rho_fV}{m}\right)-\frac{k}{m}v^n")
                    
                    # Vitesse limite
                    st.latex(r"v_{lim} = \left(\frac{mg'}{k}\right)^{1/n} = " + f"{v_lim:.4f} \text{{ m/s}}")
            
            with tab3:
                st.markdown("### Données brutes de la simulation")
                
                # Affichage des données dans un tableau avec possibilité de télécharger
                data_points = min(100, len(t_fine))  # Limiter à 100 points pour l'affichage
                step = max(1, len(t_fine) // data_points)
                
                data = {
                    "Temps (s)": t_fine[::step],
                    "Vitesse avec frottement (m/s)": v_fine[::step],
                }
                
                if show_no_drag:
                    data["Vitesse sans frottement (m/s)"] = v_no_drag[::step]
                
                # Calculer le pourcentage de la vitesse limite
                data["% de la vitesse limite"] = [v/v_lim*100 for v in v_fine[::step]]
                
                # Option pour télécharger les données complètes
                from io import StringIO
                csv_buffer = StringIO()
                
                # Créer un DataFrame complet (toutes les données)
                full_data = {
                    "Temps (s)": t_fine,
                    "Vitesse avec frottement (m/s)": v_fine,
                }
                
                if show_no_drag:
                    full_data["Vitesse sans frottement (m/s)"] = v_no_drag
                
                full_data["% de la vitesse limite"] = [v/v_lim*100 for v in v_fine]
                
                df_full = pd.DataFrame(full_data)
                df_full.to_csv(csv_buffer, index=False)
                csv_str = csv_buffer.getvalue()
                
                # Afficher un aperçu des données
                st.dataframe(pd.DataFrame(data).head(100))
                
                # Bouton pour télécharger les données complètes
                st.download_button(
                    label="Télécharger toutes les données (CSV)",
                    data=csv_str,
                    file_name="chute_verticale_data.csv",
                    mime="text/csv",
                )
        else:
            st.markdown("### Corps flottant")
            st.markdown("Lorsque la poussée d'Archimède dépasse le poids (ρ_fluide > ρ_corps), le corps flotte.")
            st.markdown("Dans ce cas, il n'y a pas de vitesse limite de chute, car le mouvement net est vers le haut.")
            
            if v0 > 0:
                st.markdown("Avec une vitesse initiale vers le bas, le corps ralentit jusqu'à s'arrêter puis remonte.")