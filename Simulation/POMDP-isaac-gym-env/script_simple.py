from tasks.simple_task import POMDPSimpleTask


def main():
    task = POMDPSimpleTask()

    while True:
        task.step()
        task.render()


if __name__=="__main__":
    main()
