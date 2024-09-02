from pystring import StringMaker
import matplotlib.pyplot as plt

stm = StringMaker("examples/in/einstein.jpg")

stm.resize(resolution=600)
stm.make_nails(n_nails=300)

stm.run(n_iter=2000, brightness_reduction_factor=0.5)

fig, ax = stm.plot_results()
fig.savefig("examples/out/einstein_results.png")

stm.seq_to_png(resolution=3000, name="examples/out/einstein.png")
stm.seq_to_movie(name="examples/out/einstein.mp4", fps=30, codec="mp4v")


plt.show()
