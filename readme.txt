Istruzioni materiale tesi

Dipendenze: 
    Le uniche dipendeze di cui dovresti avere bisogno per la prima parte della tesi sono 
        - Tensorflow (ultima versione va benissimo, quindi un semplice pip install tensorflow)
        - Numpy
        - cmake per la parte in C

    Successivamente per la parte cuda e cupy ti daremo l'accesso ad un nostro pc del lab, ma per il momento puoi tranquillamente lavorare in locale.

Organizzazione cartelle:
    - create_nnet.py: questo script ti permette di creare una rete in .h5 e di convertirla in modo automatico in .nnet (formato che richiede il codice C)
    In questo script puoi settare i vari parametri della rete e se vuoi mettere dei pesi "a mano" per eseguire dei conti su tuoi esempi o per fare debugging.

    - cartella symbolic: contiene il codice C multidimensionale della symbolic propagation. Per farlo partire modifica il path assoluto dentro main_new.c con il path della rete .nnet dentro la cartella my_nnet e cancella il contenuto della cartella build. Successivamente
    fai da dentro la cartella build: 
        - cmake ..
        - make
        - ./main

    dovresti vedere che funziona tutto. Questa rete è la stessa che abbiamo visto ieri nella presentazione che ritorna intervallo [6 16]. Il file main_new.c ha 
    già tutto in un unico file, cosi puoi studiarlo agevolmente. Come dicevamo ieri il tuo compito è trasformare il codice multidimensionale in monodimensionale. O in alternativa, trovare
    un modo alternativo per passare da un vettore monodimensionale di pesi ad una forma multidimensionale nel kernel (vedi punto sotto)

    - cartella cuda_integration: qui trovi due file. In particolare il file cuda_code.py è il kernel C della naive interval propagation che viene eseguito in parallelo su GPU. Mentre il file test_propGPU.py contiene il mail che chiama
    la propagation su GPU degli intervalli. Per runnare questo script ti servirebbe una GPU e cupy. Se non hai a disposizione queste cose intanto puoi solo vedere in che formato arrivano i pesi della rete e gli intervalli e come vengono letti nel kernel per farti un'idea più precisa.


Il codice originale di ProVe, i.e., il nostro verificatore lo trovi qui https://github.com/d-corsi/NetworkVerifier
Per qualsiasi dubbio scrivimi pure una mail  