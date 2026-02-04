from matplotlib.animation import FuncAnimation
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

def rk4(f, y0, t):
    y = np.zeros((len(t), len(y0)))
    y[0] = y0
    for i in range(len(t) - 1):
        h = t[i+1] - t[i]
        k1 = f(t[i], y[i])
        k2 = f(t[i] + h/2, y[i] + h/2 * k1)
        k3 = f(t[i] + h/2, y[i] + h/2 * k2)
        k4 = f(t[i] + h, y[i] + h * k3)
        y[i+1] = y[i] + h/6 * (k1 + 2*k2 + 2*k3 + k4)
    return y

G = 1.0
m1 = m2 = m3 = 1.0

def f(t, y):
    # Θέσεις
    r1 = y[0:3]
    r2 = y[3:6]
    r3 = y[6:9]

    # Ταχύτητες
    v1 = y[9:12]
    v2 = y[12:15]
    v3 = y[15:18]

    # Διανύσματα απόστασης
    r12 = r2 - r1
    r13 = r3 - r1
    r23 = r3 - r2

    # Μέτρα
    d12 = np.linalg.norm(r12)
    d13 = np.linalg.norm(r13)
    d23 = np.linalg.norm(r23)

    # Επιταχύνσεις
    a1 = G * (m2 * r12 / d12**3 + m3 * r13 / d13**3)
    a2 = G * (m1 * (-r12) / d12**3 + m3 * r23 / d23**3)
    a3 = G * (m1 * (-r13) / d13**3 + m2 * (-r23) / d23**3)

    # dy/dt
    dydt = np.zeros_like(y)
    # dr/dt = v
    dydt[0:3] = v1
    dydt[3:6] = v2
    dydt[6:9] = v3
    # dv/dt = a
    dydt[9:12] = a1
    dydt[12:15] = a2
    dydt[15:18] = a3

    return dydt

r1_0 = np.array([1.55,  0.3, 0.0])
r2_0 = np.array([ 3.5,  0.3, 0.0])
r3_0 = np.array([ 3,  0.3, 0.0])

v1_0 = np.array([0.4,  0.8, 0.0])
v2_0 = np.array([0,    1.5, 0.0])
v3_0 = np.array([0,  0.2,  0.0])


y0 = np.hstack([r1_0, r2_0, r3_0,
                v1_0, v2_0, v3_0])

T = 3.5
h = 0.01
t = np.arange(0.0, T + h, h)

y = rk4(f, y0, t)

# --- Προετοιμασία του Γραφήματος ---
fig, ax = plt.subplots(figsize=(8, 8))
ax.set_aspect('equal')
ax.grid(True, linestyle='--', alpha=0.5)
ax.set_xlim(0, 5)  # Προάρμοσε τα όρια αν τα σώματα βγαίνουν έξω
ax.set_ylim(0, 5)

# Δημιουργία των αντικειμένων που θα κινούνται
line1, = ax.plot([], [], 'r-', lw=1, alpha=0.6, label="Body 1")
line2, = ax.plot([], [], 'g-', lw=1, alpha=0.6, label="Body 2")
line3, = ax.plot([], [], 'b-', lw=1, alpha=0.6, label="Body 3")

dot1, = ax.plot([], [], 'ro', markersize=8)
dot2, = ax.plot([], [], 'go', markersize=8)
dot3, = ax.plot([], [], 'bo', markersize=8)

ax.legend()

# --- Συνάρτηση Ενημέρωσης των Frames ---
def update(frame):
    # Ενημέρωση γραμμών τροχιάς
    line1.set_data(y[:frame, 0], y[:frame, 1])
    line2.set_data(y[:frame, 3], y[:frame, 4])
    line3.set_data(y[:frame, 6], y[:frame, 7])
    
    # Ενημέρωση τρεχουσών θέσεων (τελείες)
    dot1.set_data([y[frame, 0]], [y[frame, 1]])
    dot2.set_data([y[frame, 3]], [y[frame, 4]])
    dot3.set_data([y[frame, 6]], [y[frame, 7]])
    
    return line1, line2, line3, dot1, dot2, dot3

# --- Δημιουργία και Αποθήκευση ---
# Δημιουργούμε το animation
ani = animation.FuncAnimation(fig, update, frames=len(t), interval=20, blit=True)

# ΑΥΤΗ Η ΓΡΑΜΜΗ ΤΟ ΣΩΖΕΙ ΩΣ GIF:
print("Αποθήκευση animation... παρακαλώ περιμένετε.")
ani.save("three_body_simulation.gif", writer='pillow', fps=30)
print("Έτοιμο! Το αρχείο 'three_body_simulation.gif' βρίσκεται στον φάκελό σου.")

plt.show()
