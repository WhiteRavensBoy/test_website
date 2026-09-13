from locust import HttpUser, task, between

class MyUser(HttpUser):
    wait_time = between(1,3)

    @task
    def hello_world(self):
        self.client.get(f"/ask/who%20am%20i")
        
