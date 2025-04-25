
без картинок не воспринимается..

знаете, я совершенно не хотел разбирать эту статью.. 0 картинок или графиков, 4 страницы формул и нет текстового слоя в пдф.. вроде все всё знают, ну что там обратное распространение: есть forward - просто подсчёт значения, есть backward - по chain rule обратно прокидываем градиенты... а оказывается это просто кто-то придумал и долго никто не доказывал, а потом вот чел доказал через Лагранжа и всё сошлось! по моему забавно и постфактум - статья стоит прочтения, ну скорее не статья а вот эти заметки о статье, всё таки и читается проще и суть передаёт понятнее


1989 – 1998: на базе этой теории появляются LeNet-1/LeNet-5, алгоритмы ускорения (Becker & LeCun, «Efficient BackProp») и конечно вся волна свёрточных сетей.

2000-е: фреймворки автодифференцирования (Theano → TensorFlow/PyTorch) реализуют ровно тот же принцип reverse-mode, описанный в 1988-м.

Современные методы (Transformers, Diffusion) по-прежнему опираются на этот «двойной проход».




> Обучение сети формулируется как задача минимизации функции ошибки  $E(\mathbf x^{L}, \mathbf y)$  
>  при **жёстких ограничениях** — уравнениях прямого прохода каждого слоя.  
> Предлагается избавиться от ограничений применяя метод Лагранжа.

---

### 1. Формулируем задачу 

Пусть сеть имеет $L$ слоёв. Для каждого слоя $l=1,\dots ,L$

$$
\mathbf x^{l}=f^{l}(\mathbf x^{\,l-1},\;\mathbf w^{l}) ,
$$

где $\mathbf x^{0}=\mathbf x_{\text{in}}$ — вход, $\mathbf w^{l}$ — обучаемые веса слоя, а $\mathbf x^{L}$ — выход сети.

Перепишем: 

$$
F^{l}(\mathbf x^{l},\mathbf x^{\,l-1},\mathbf w^{l}) \;=\;  
\mathbf x^{l}-f^{l}(\mathbf x^{\,l-1},\mathbf w^{l}) \;=\; 0.
$$

---

### 2. Лагранжиан  

$$
\mathcal L(\mathbf x,\mathbf w,\boldsymbol\lambda)\;=\;
E\!\left(\mathbf x^{L},\mathbf y\right)\;+\;
\sum_{l=1}^{L}\;(\boldsymbol\lambda^{l})^{\!\top}
\bigl[\mathbf x^{l}-f^{l}(\mathbf x^{\,l-1},\mathbf w^{l})\bigr].
\tag{1}
$$

Получили задачу $\min_{\mathbf w,\mathbf x}\max_{\boldsymbol\lambda}\mathcal L$ без ограничений ($\boldsymbol\lambda^{l}\in\mathbb R^{\dim(\mathbf x^{l})}$).   
Чтобы решить дальше берём производную и приравниваем к 0.

---

### 3. Получаемые условия   


$$
\frac{\partial\mathcal L}{\partial \mathbf x^{L}}
\;=\;
\frac{\partial E}{\partial \mathbf x^{L}}
\;+\;
\boldsymbol\lambda^{L}
\;=\;0
\;\Longrightarrow\;
\boxed{\boldsymbol\lambda^{L}=-\frac{\partial E}{\partial\mathbf x^{L}}}.
\tag{2}
$$

Это ошибка на выходе слоя.


Для любого $l=L-1,\dots ,1$:

$$
\frac{\partial\mathcal L}{\partial \mathbf x^{l}}
=\boldsymbol\lambda^{l}
-\Bigl(\frac{\partial f^{l+1}}{\partial\mathbf x^{l}}\Bigr)^{\!\top}\!
\boldsymbol\lambda^{l+1}=0
\Longrightarrow
\boxed{\boldsymbol\lambda^{l}
=\Bigl(\frac{\partial f^{l+1}}{\partial\mathbf x^{l}}\Bigr)^{\!\top}
\boldsymbol\lambda^{l+1}}.
\tag{3}
$$

Получили рекуррентная формула ровно такую же как в back-propogation!


---

### 4. Производная по весам = градиент сети  

$$
\frac{\partial\mathcal L}{\partial \mathbf w^{l}}
=-\bigl(\boldsymbol\lambda^{l}\bigr)^{\!\top}
\frac{\partial f^{l}}{\partial \mathbf w^{l}}
\;\equiv\;
-\frac{\partial E}{\partial \mathbf w^{l}}.
\tag{4}
$$

Знак минус компенсируется в шаге спуска, то есть  
**$\nabla_{\!\mathbf w^{l}}E$** — это внутренняя часть в backprop.

---

### 5. Что нам это дало  

| Вывод | Интерпретация |
|-------|---------------|
| λ-векторы ≡ δ-ошибки | «Ошибка» — это множитель Лагранжа у равенств слоя. |
| (2)–(3) = back-pass | Правила вычисления λ совпали с BP; получили **reverse-mode AD** как следствие лагранжевой стационарности. |
| (4) = градиент по весам | После одного обратного прохода имеем ∇E по всем параметрам. |
| Сложность | $O(2·\|E\|)$ | Один прямой + один обратный проход по тем же рёбрам графа  |

