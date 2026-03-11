def limpar_texto(texto: str) -> str:
    """
    Limpeza básica do texto de sintomas.

    Args:
        texto: Texto livre descrevendo sintomas do paciente.

    Returns:
        Texto normalizado pronto para vetorização.
    """
    texto = str(texto).lower().strip()
    texto = texto.replace(",", " ")
    return " ".join(texto.split())  # remove espaços extras








