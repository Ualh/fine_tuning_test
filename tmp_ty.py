import typer

app = typer.Typer()

@app.command()
def main(config: str = typer.Option('config.yaml', '--config', '-c')):
    pass

if __name__ == '__main__':
    app()
