from pystring import StringMaker
import matplotlib.pyplot as plt

stm = StringMaker("examples/in/einstein.jpg")

stm.resize(resolution=400)
stm.make_nails(n_nails=300, mode="combined")

stm.run(n_iter=3000, brightness_reduction_factor=0.7)

fig, ax = stm.plot_results()
fig.savefig("examples/out/out_diag.png")

stm.seq_to_png(resolution=3000, name="examples/out/out.png")
stm.seq_to_movie(name="examples/out/out.mp4", fps=30, codec="mp4v")


plt.show()
