from grpc_wrapper.client import create_client


def run():
    client = create_client(ip="사용자 지정 아이피", port="사용자 지정 포트")
    print("초기화 우선 실행")
    print(client.send({"query":"일상 대화 초기화"}))
    while True:
        string = input("입력:")
        input_ = {
            "query": string
        }
        output = client.send(input_ )
        print(output)


if __name__ == "__main__":
    run()
