# Zadatak 4 — L2 regularizacija polinomijalne regresije

## Model i funkcija cilja

Polinomijalni model ostaje

$$
\hat y(x;\mathbf w)=\sum_{j=0}^{M}w_j x^j,
\qquad \Phi_{ij}=x_i^j.
$$

Za Ridge regularizaciju uzimamo

$$
\boxed{E_\lambda(\mathbf w)=\frac{1}{2N}\|\mathbf y-\Phi\mathbf w\|_2^2
+\lambda\sum_{j=1}^{M}w_j^2},\quad\lambda\ge0.
$$

Koeficijent $w_0$ ne regularizujemo; kaznjavaju se ostali koeficijenti.
Ovaj izbor funkcije cilja je konzistentan sa implementacijom.

## Parcijalni izvodi

Neka je $D=\mathrm{diag}(0,1,\ldots,1)$. Tada

$$
E_\lambda(\mathbf w)=\frac{1}{2N}(\mathbf y-\Phi\mathbf w)^T
(\mathbf y-\Phi\mathbf w)+\lambda\mathbf w^TD\mathbf w.
$$

Gradijent po $\mathbf w$ je

$$
\nabla E_\lambda=-\frac1N\Phi^T(\mathbf y-\Phi\mathbf w)
+2\lambda D\mathbf w.
$$

Postavljanjem gradijenta na nulu i mnozenjem sa N:

$$
\Phi^T\Phi\mathbf w-\Phi^T\mathbf y+2N\lambda D\mathbf w=0.
$$

Zato su regularizovane normalne jednacine

$$
\boxed{(\Phi^T\Phi+2N\lambda D)\mathbf w=\Phi^T\mathbf y}.
$$

Kada je leva matrica invertibilna, resenje je

$$
\boxed{\mathbf w=(\Phi^T\Phi+2N\lambda D)^{-1}\Phi^T\mathbf y}.
$$

U implementaciji se koristi `solve`, odnosno kod numericki zahtevnih
polinoma isti sistem se resava sa vise decimalnih cifara.

**Zasto faktor $2N$?** Zato sto je ispred rezidualne sume faktor
$1/(2N)$, a regularizacioni clan je zapisan kao
$\lambda\sum_{j=1}^Mw_j^2$. Ako se izabere drugacija konvencija za
koeficijent regularizacije, menja se i faktor uz $\lambda$ u sistemu.

## Poredjenje

- $\lambda=0$: obicna polinomijalna regresija, bez regularizacije.
- $\lambda>0$: kaznjavaju se veci koeficijenti, sto ogranicava
  kompleksnost modela i cesto smanjuje oscilacije van obucavajuceg uzorka.
- Veca $\lambda$ ne mora znaciti manju MSE na obucavajucim podacima:
  dodavanje kazne obicno povecava rezidualnu gresku, ali moze pomoci
  generalizaciji.
- Zbog direktne upotrebe padavina u mm, stepeni $x^j$ imaju znatno
  razlicite skale. Zbog toga efekat istog $\lambda$ nije jednak na sve
  koeficijente i interpretacija je osetljiva na jedinice.
