"""
Math Visualizer - MVP for A-Level Further Maths / STEP Students
================================================================
A minimal Streamlit app providing interactive visualizations for core Further Maths topics.
Designed as an MVP that can be extended with additional features, more sophisticated
numerical methods, and expanded topic coverage.

Author: Generated for Cambridge application portfolio
Target audience: A-Level Further Maths and STEP preparation students
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import cmath

# Page configuration
st.set_page_config(
    page_title="Math Visualizer",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better mobile compatibility
st.markdown("""
    <style>
    .stSlider > div > div > div > div {
        font-size: 0.8rem;
    }
    </style>
""", unsafe_allow_html=True)


def matrix_transformation_visualizer():
    """
    Matrix Transformation Visualizer
    Displays 2D linear transformations with animation and geometric properties
    """
    st.title("🔲 Matrix Transformation Visualizer")
    st.markdown("Explore 2×2 matrix transformations on the plane")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Matrix Entries")
        a11 = st.number_input("a₁₁ (top-left)", value=1.0, step=0.1)
        a12 = st.number_input("a₁₂ (top-right)", value=0.0, step=0.1)
        a21 = st.number_input("a₂₁ (bottom-left)", value=0.0, step=0.1)
        a22 = st.number_input("a₂₂ (bottom-right)", value=1.0, step=0.1)
        
        # Interpolation slider
        t = st.slider("Interpolation t (0 = original, 1 = transformed)", 
                      min_value=0.0, max_value=1.0, value=1.0, step=0.05)
        
        # Matrix properties
        matrix = np.array([[a11, a12], [a21, a22]])
        det = np.linalg.det(matrix)
        
        st.subheader("Matrix Properties")
        st.write(f"**Determinant:** {det:.3f}")
        st.write(f"**Invertible:** {'Yes' if abs(det) > 1e-10 else 'No'}")
        st.write(f"**Area scaling factor:** {abs(det):.3f}")
        
        # Eigenvalues and eigenvectors
        try:
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            st.subheader("Eigenvalues")
            for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
                if np.isreal(val) and abs(np.imag(val)) < 1e-10:
                    st.write(f"λ{i+1} = {np.real(val):.3f}")
                    st.write(f"v{i+1} = [{vec[0].real:.2f}, {vec[1].real:.2f}]")
        except:
            st.write("Cannot compute eigenvalues")
    
    with col2:
        # Define original shape (unit square + additional points for visualization)
        square = np.array([
            [0, 0], [1, 0], [1, 1], [0, 1], [0, 0]  # Unit square
        ]).T
        
        # Additional sample points
        extra_points = np.array([
            [0.5, 0], [1, 0.5], [0.5, 1], [0, 0.5],  # Midpoints
            [0.5, 0.5]  # Center
        ]).T
        
        # Apply linear interpolation: (1-t)*I + t*A
        identity = np.eye(2)
        interpolated_matrix = (1 - t) * identity + t * matrix
        
        # Transform shapes
        transformed_square = interpolated_matrix @ square
        transformed_extra = interpolated_matrix @ extra_points
        
        # Create plot
        fig = go.Figure()
        
        # Original square (light gray)
        fig.add_trace(go.Scatter(
            x=square[0], y=square[1],
            mode='lines+markers',
            name='Original',
            line=dict(color='lightgray', dash='dash'),
            marker=dict(size=6)
        ))
        
        # Transformed square
        fig.add_trace(go.Scatter(
            x=transformed_square[0], y=transformed_square[1],
            mode='lines+markers',
            name='Transformed',
            line=dict(color='blue', width=2),
            marker=dict(size=8)
        ))
        
        # Extra points
        fig.add_trace(go.Scatter(
            x=transformed_extra[0], y=transformed_extra[1],
            mode='markers',
            name='Sample points',
            marker=dict(size=10, color='red', symbol='x')
        ))
        
        # Add eigenvectors if real
        try:
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            for i, (val, vec) in enumerate(zip(eigenvalues, eigenvectors.T)):
                if np.isreal(val) and abs(np.imag(val)) < 1e-10:
                    vec_real = vec.real
                    scale = 1.5
                    fig.add_trace(go.Scatter(
                        x=[0, scale * vec_real[0]],
                        y=[0, scale * vec_real[1]],
                        mode='lines',
                        name=f'Eigenvector {i+1}',
                        line=dict(width=3, dash='dot')
                    ))
        except:
            pass
        
        # Layout with equal aspect ratio
        fig.update_layout(
            title=f"Transformation (t={t:.2f})",
            xaxis=dict(title="x", scaleanchor="y", scaleratio=1, zeroline=True),
            yaxis=dict(title="y", zeroline=True),
            width=700,
            height=700,
            showlegend=True,
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)


def complex_number_plotter():
    """
    Complex Number Plotter - Argand diagram with operations
    """
    st.title("🔢 Complex Number Plotter")
    st.markdown("Visualize complex numbers on the Argand plane")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Input Complex Number")
        input_method = st.radio("Input method:", ["Separate components", "a+bi notation"])
        
        if input_method == "Separate components":
            real = st.number_input("Real part", value=1.0, step=0.1)
            imag = st.number_input("Imaginary part", value=1.0, step=0.1)
        else:
            complex_str = st.text_input("Enter complex number (e.g., 3+4j or 2-1j)", value="1+1j")
            try:
                z = complex(complex_str.replace('i', 'j'))
                real, imag = z.real, z.imag
            except:
                st.error("Invalid format. Using default 1+1j")
                real, imag = 1.0, 1.0
        
        z = complex(real, imag)
        modulus = abs(z)
        argument = cmath.phase(z)
        
        st.subheader("Properties")
        st.write(f"**z = {real:.3f} + {imag:.3f}i**")
        st.write(f"**Modulus |z| = {modulus:.3f}**")
        st.write(f"**Argument arg(z) = {argument:.3f} rad** ({np.degrees(argument):.1f}°)")
        
        st.subheader("Multiply by:")
        mult_real = st.number_input("Real part (multiplier)", value=1.0, step=0.1, key="mult_real")
        mult_imag = st.number_input("Imag part (multiplier)", value=0.0, step=0.1, key="mult_imag")
        w = complex(mult_real, mult_imag)
        result = z * w
        
        st.write(f"**Result: {result.real:.3f} + {result.imag:.3f}i**")
        st.write(f"**Modulus: {abs(result):.3f}**")
        st.write(f"**Argument: {cmath.phase(result):.3f} rad**")
        
        st.subheader("Roots of Unity")
        show_roots = st.checkbox("Show nth roots of unity")
        if show_roots:
            n = st.number_input("n (number of roots)", min_value=2, max_value=20, value=5, step=1)
    
    with col2:
        fig = go.Figure()
        
        # Plot original complex number
        fig.add_trace(go.Scatter(
            x=[0, real], y=[0, imag],
            mode='lines+markers',
            name='z',
            line=dict(color='blue', width=3),
            marker=dict(size=12),
            hovertemplate='z = %{x:.3f} + %{y:.3f}i<br>|z| = ' + f'{modulus:.3f}<br>arg(z) = {argument:.3f}'
        ))
        
        # Plot result of multiplication
        fig.add_trace(go.Scatter(
            x=[0, result.real], y=[0, result.imag],
            mode='lines+markers',
            name='z × w',
            line=dict(color='red', width=3, dash='dash'),
            marker=dict(size=12, symbol='square'),
            hovertemplate='z×w = %{x:.3f} + %{y:.3f}i'
        ))
        
        # Roots of unity
        if show_roots:
            roots = [cmath.exp(2j * cmath.pi * k / n) for k in range(n)]
            roots_real = [r.real for r in roots]
            roots_imag = [r.imag for r in roots]
            
            fig.add_trace(go.Scatter(
                x=roots_real, y=roots_imag,
                mode='markers',
                name=f'{n}th roots of unity',
                marker=dict(size=10, color='green', symbol='diamond'),
                hovertemplate='%{x:.3f} + %{y:.3f}i'
            ))
            
            # Unit circle
            theta = np.linspace(0, 2*np.pi, 100)
            fig.add_trace(go.Scatter(
                x=np.cos(theta), y=np.sin(theta),
                mode='lines',
                name='Unit circle',
                line=dict(color='lightgray', dash='dot'),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Layout
        max_val = max(abs(real), abs(imag), abs(result.real), abs(result.imag), 2)
        fig.update_layout(
            title="Argand Diagram",
            xaxis=dict(title="Real axis", scaleanchor="y", scaleratio=1, 
                      zeroline=True, range=[-max_val*1.2, max_val*1.2], gridcolor='lightgray'),
            yaxis=dict(title="Imaginary axis", zeroline=True, 
                      range=[-max_val*1.2, max_val*1.2], gridcolor='lightgray'),
            width=700,
            height=700,
            showlegend=True,
            hovermode='closest'
        )
        
        st.plotly_chart(fig, use_container_width=True)


def graph_transformation_explorer():
    """
    Graph Transformation Explorer - y = a*f(bx + c) + d
    """
    st.title("📈 Graph Transformation Explorer")
    st.markdown("Explore transformations: **y = a·f(b·x + c) + d**")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Base Function")
        function_name = st.selectbox(
            "Select f(x):",
            ["sin(x)", "cos(x)", "tan(x)", "exp(x)", "x²", "x³", "1/x"]
        )
        
        st.subheader("Transformation Parameters")
        a = st.slider("a (vertical stretch/reflection)", -5.0, 5.0, 1.0, 0.1)
        b = st.slider("b (horizontal stretch/reflection)", -5.0, 5.0, 1.0, 0.1)
        c = st.slider("c (horizontal shift, left if +)", -10.0, 10.0, 0.0, 0.1)
        d = st.slider("d (vertical shift)", -10.0, 10.0, 0.0, 0.1)
        
        st.subheader("x-axis Range")
        x_min = st.number_input("x min", value=-10.0, step=1.0)
        x_max = st.number_input("x max", value=10.0, step=1.0)
        
        if function_name in ["sin(x)", "cos(x)", "tan(x)"]:
            show_inverse = st.checkbox("Show inverse function")
        else:
            show_inverse = False
    
    with col2:
        # Generate x values
        x = np.linspace(x_min, x_max, 1000)
        
        # Define base functions
        functions = {
            "sin(x)": lambda x: np.sin(x),
            "cos(x)": lambda x: np.cos(x),
            "tan(x)": lambda x: np.tan(x),
            "exp(x)": lambda x: np.exp(x),
            "x²": lambda x: x**2,
            "x³": lambda x: x**3,
            "1/x": lambda x: 1/x
        }
        
        base_func = functions[function_name]
        
        # Apply transformation: y = a*f(bx + c) + d
        # Handle division by zero and domain issues
        try:
            if b != 0:
                y = a * base_func(b * x + c) + d
            else:
                y = np.full_like(x, d)
            
            # Clip extreme values for better visualization
            y = np.clip(y, -100, 100)
        except:
            y = np.zeros_like(x)
        
        fig = go.Figure()
        
        # Original function (light)
        try:
            y_original = base_func(x)
            y_original = np.clip(y_original, -100, 100)
            fig.add_trace(go.Scatter(
                x=x, y=y_original,
                mode='lines',
                name=f'Original: {function_name}',
                line=dict(color='lightgray', dash='dash'),
                opacity=0.5
            ))
        except:
            pass
        
        # Transformed function
        fig.add_trace(go.Scatter(
            x=x, y=y,
            mode='lines',
            name=f'y = {a:.1f}·f({b:.1f}x + {c:.1f}) + {d:.1f}',
            line=dict(color='blue', width=3)
        ))
        
        # Inverse function (for trig)
        if show_inverse:
            inverse_funcs = {
                "sin(x)": np.arcsin,
                "cos(x)": np.arccos,
                "tan(x)": np.arctan
            }
            if function_name in inverse_funcs:
                # Inverse on restricted domain
                x_inv = np.linspace(-1, 1, 1000) if function_name != "tan(x)" else x
                try:
                    y_inv = inverse_funcs[function_name](x_inv)
                    fig.add_trace(go.Scatter(
                        x=x_inv, y=y_inv,
                        mode='lines',
                        name=f'Inverse: {function_name}⁻¹',
                        line=dict(color='red', dash='dot', width=2)
                    ))
                except:
                    pass
        
        # Mark special points (roots, extrema)
        # Simple root finding for zeros
        if len(y) > 0:
            zero_crossings = np.where(np.diff(np.sign(y)))[0]
            if len(zero_crossings) > 0 and len(zero_crossings) < 20:
                roots_x = x[zero_crossings]
                roots_y = y[zero_crossings]
                fig.add_trace(go.Scatter(
                    x=roots_x, y=roots_y,
                    mode='markers',
                    name='Approximate roots',
                    marker=dict(size=10, color='red', symbol='x')
                ))
        
        fig.update_layout(
            title=f"Transformed {function_name}",
            xaxis=dict(title="x", zeroline=True, gridcolor='lightgray'),
            yaxis=dict(title="y", zeroline=True, gridcolor='lightgray'),
            width=700,
            height=600,
            showlegend=True,
            hovermode='x'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.info(f"""
        **Transformation guide:**
        - **a**: vertical stretch (|a|>1) or compression (|a|<1); reflection if a<0
        - **b**: horizontal compression (|b|>1) or stretch (|b|<1); reflection if b<0
        - **c**: horizontal shift LEFT by c/b units
        - **d**: vertical shift UP by d units
        """)


def topic_index():
    """
    Topic Index for Pearson Edexcel AS/A-Level Further Maths Core Pure
    """
    st.title("📚 Topic Index: Core Pure Mathematics")
    st.markdown("Pearson Edexcel AS/A-Level Further Mathematics - Visualization Ideas")
    
    topics = [
        {
            "topic": "Complex Numbers",
            "viz": "Argand diagram showing operations (multiplication, division, powers) and loci in the complex plane."
        },
        {
            "topic": "Matrices and Transformations",
            "viz": "2D/3D transformation visualizer showing rotation, reflection, scaling, shear with eigenvalue decomposition."
        },
        {
            "topic": "Vectors",
            "viz": "3D vector operations, scalar/vector products, projections, and planes in space with interactive manipulation."
        },
        {
            "topic": "Polynomial Equations & Roots",
            "viz": "Graph of polynomial with roots, turning points, and behavior at infinity; show relationship between coefficients and roots."
        },
        {
            "topic": "Series (Arithmetic, Geometric, Power)",
            "viz": "Animated partial sums converging to series limit; interactive sigma notation builder showing pattern recognition."
        },
        {
            "topic": "Calculus: Differentiation",
            "viz": "Tangent line animation at point on curve; gradient function overlay; visual proof of standard derivatives."
        },
        {
            "topic": "Calculus: Integration",
            "viz": "Riemann sum rectangles converging to area under curve; animated demonstration of fundamental theorem."
        },
        {
            "topic": "Numerical Methods",
            "viz": "Newton-Raphson iteration visualization; trapezium rule area approximation with error bounds."
        },
        {
            "topic": "Trigonometric Graphs & Identities",
            "viz": "Interactive trig functions with amplitude/period/phase sliders; unit circle animation showing angle relationships."
        },
        {
            "topic": "Hyperbolic Functions",
            "viz": "Graphs of sinh, cosh, tanh with geometric interpretation using hyperbola; comparison to trig functions."
        },
        {
            "topic": "Differential Equations (1st Order)",
            "viz": "Slope field visualization with solution curves; interactive initial value problem solver showing family of solutions."
        },
        {
            "topic": "Coordinate Geometry (Conics)",
            "viz": "Interactive parabola, ellipse, hyperbola with focus, directrix, and eccentricity relationships shown visually."
        },
        {
            "topic": "Further Algebra (Binomial Theorem)",
            "viz": "Pascal's triangle builder; animated expansion showing term generation; coefficient pattern exploration."
        },
        {
            "topic": "STEP-style Geometry Problems",
            "viz": "Dynamic geometry constructions for common STEP problems (circles, triangles, loci) with draggable points."
        },
        {
            "topic": "Proof Techniques",
            "viz": "Visual proof demonstrations (e.g., sum of squares, geometric series) with step-by-step animated construction."
        }
    ]
    
    for i, item in enumerate(topics, 1):
        with st.expander(f"{i}. {item['topic']}", expanded=False):
            st.write(f"**Suggested visualization:** {item['viz']}")
    
    st.markdown("---")
    st.info("""
    **Note:** This app currently implements visualizations for Complex Numbers, Matrices, and Graph Transformations.
    The remaining topics are listed here as a roadmap for future development. Each would benefit from interactive
    visualizations to build intuition and understanding for exam preparation.
    """)


def extras_page():
    """
    Extras - Small additional interactive demos
    """
    st.title("🎁 Extras: Bonus Visualizations")
    
    demo_choice = st.selectbox(
        "Select a demo:",
        ["Riemann Sum Integration", "Trigonometric Function Gallery"]
    )
    
    if demo_choice == "Riemann Sum Integration":
        st.subheader("Riemann Sum vs Definite Integral")
        st.markdown("Visualize how rectangles approximate area under a curve")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            func_choice = st.selectbox("Function:", ["x²", "sin(x)", "exp(x)", "1/(1+x²)"])
            a = st.number_input("Lower bound a", value=0.0, step=0.1)
            b = st.number_input("Upper bound b", value=2.0, step=0.1)
            n = st.slider("Number of rectangles", min_value=1, max_value=100, value=10)
            method = st.radio("Method:", ["Left Riemann", "Right Riemann", "Midpoint", "Trapezoid"])
        
        with col2:
            functions = {
                "x²": lambda x: x**2,
                "sin(x)": lambda x: np.sin(x),
                "exp(x)": lambda x: np.exp(x),
                "1/(1+x²)": lambda x: 1/(1+x**2)
            }
            
            f = functions[func_choice]
            x_smooth = np.linspace(a, b, 500)
            y_smooth = f(x_smooth)
            
            # Calculate Riemann sum
            dx = (b - a) / n
            
            if method == "Left Riemann":
                x_sample = np.linspace(a, b - dx, n)
            elif method == "Right Riemann":
                x_sample = np.linspace(a + dx, b, n)
            elif method == "Midpoint":
                x_sample = np.linspace(a + dx/2, b - dx/2, n)
            else:  # Trapezoid
                x_sample = np.linspace(a, b, n+1)
            
            y_sample = f(x_sample)
            
            if method == "Trapezoid":
                riemann_sum = np.trapz(y_sample, x_sample)
            else:
                riemann_sum = np.sum(y_sample) * dx
            
            fig = go.Figure()
            
            # Plot function
            fig.add_trace(go.Scatter(
                x=x_smooth, y=y_smooth,
                mode='lines',
                name=f'y = {func_choice}',
                line=dict(color='blue', width=3)
            ))
            
            # Plot rectangles
            if method != "Trapezoid":
                for i in range(n):
                    x_rect = [a + i*dx, a + (i+1)*dx, a + (i+1)*dx, a + i*dx, a + i*dx]
                    y_rect = [0, 0, y_sample[i], y_sample[i], 0]
                    fig.add_trace(go.Scatter(
                        x=x_rect, y=y_rect,
                        fill='toself',
                        fillcolor='rgba(255, 0, 0, 0.3)',
                        line=dict(color='red', width=1),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
            else:
                # Trapezoids
                for i in range(n):
                    x_trap = [x_sample[i], x_sample[i+1], x_sample[i+1], x_sample[i], x_sample[i]]
                    y_trap = [0, 0, y_sample[i+1], y_sample[i], 0]
                    fig.add_trace(go.Scatter(
                        x=x_trap, y=y_trap,
                        fill='toself',
                        fillcolor='rgba(255, 0, 0, 0.3)',
                        line=dict(color='red', width=1),
                        showlegend=False,
                        hoverinfo='skip'
                    ))
            
            fig.update_layout(
                title=f"{method} Sum: {riemann_sum:.6f}",
                xaxis=dict(title="x"),
                yaxis=dict(title="y"),
                width=700,
                height=500,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            st.write(f"**Approximate integral:** {riemann_sum:.6f}")
            st.write(f"**Number of rectangles:** {n}")
            st.write(f"**Width of each:** {dx:.6f}")
    
    else:  # Trigonometric Function Gallery
        st.subheader("Trigonometric Function Gallery")
        st.markdown("Explore amplitude, period, and phase shifts")
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            trig_func = st.selectbox("Function:", ["sin", "cos", "tan", "sec", "csc", "cot"])
            amplitude = st.slider("Amplitude A", 0.1, 5.0, 1.0, 0.1)
            frequency = st.slider("Frequency ω (affects period)", 0.1, 5.0, 1.0, 0.1)
            phase = st.slider("Phase shift φ (radians)", -np.pi, np.pi, 0.0, 0.1)
            vertical = st.slider("Vertical shift k", -3.0, 3.0, 0.0, 0.1)
        
        with col2:
            x = np.linspace(-2*np.pi, 2*np.pi, 1000)
            
            trig_functions = {
                "sin": np.sin,
                "cos": np.cos,
                "tan": np.tan,
                "sec": lambda x: 1/np.cos(x),
                "csc": lambda x: 1/np.sin(x),
                "cot": lambda x: 1/np.tan(x)
            }
            
            base_func = trig_functions[trig_func]
            y = amplitude * base_func(frequency * x + phase) + vertical
            y = np.clip(y, -10, 10)  # Clip for visualization
            
            fig = go.Figure()
            
            fig.add_trace(go.Scatter(
                x=x, y=y,
                mode='lines',
                name=f'y = {amplitude:.1f}{trig_func}({frequency:.1f}x + {phase:.2f}) + {vertical:.1f}',
                line=dict(color='purple', width=3)
            ))
            
            fig.update_layout(
                title=f"y = A·{trig_func}(ωx + φ) + k",
                xaxis=dict(title="x (radians)", zeroline=True),
                yaxis=dict(title="y", zeroline=True),
                width=700,
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            period = 2 * np.pi / frequency if trig_func in ["sin", "cos", "sec", "csc"] else np.pi / frequency
            st.write(f"**Period:** {period:.3f}")
            st.write(f"**Amplitude:** {amplitude:.1f}")
            st.write(f"**Phase shift:** {phase:.3f} rad ({np.degrees(phase):.1f}°)")


def main():
    """
    Main application entry point with navigation
    """
    st.sidebar.title("📐 Math Visualizer")
    st.sidebar.markdown("*A-Level Further Maths & STEP*")
    
    page = st.sidebar.radio(
        "Navigate to:",
        [
            "Matrix Transformation Visualizer",
            "Complex Number Plotter",
            "Graph Transformation Explorer",
            "Topic Index (Core Pure)",
            "Extras"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **About this app:**
    An MVP for visualizing A-Level Maths/Further Maths concepts.
    Built with Streamlit.
    
    **Target users:** A-Level Maths/Further Maths and STEP students
    """)
    
    # Route to appropriate page
    if page == "Matrix Transformation Visualizer":
        matrix_transformation_visualizer()
    elif page == "Complex Number Plotter":
        complex_number_plotter()
    elif page == "Graph Transformation Explorer":
        graph_transformation_explorer()
    elif page == "Topic Index (Core Pure)":
        topic_index()
    elif page == "Extras":
        extras_page()


if __name__ == "__main__":
    main()