import time
import os
# import openai
import sys
from parse_c_function_names import func_name_extract

# class LMFunction(object):
#     def __init__(self, engine='gpt-3.5-turbo', max_tokens=512):
#         self.engine = engine
#         self.max_tokens = max_tokens
#         self.openai = openai
#         openai.api_key = os.environ['OPENAI_API_KEY']

#     def _call_api(self, prompt, engine, max_tokens, max_retries=10, retry_wait=2):
#         for i in range(max_retries):
#             try:
#                 return self.openai.ChatCompletion.create(
#                     model=engine,
#                     messages=[
#                         {"role": "system", "content": "You are a helpful assistant."},
#                         {"role": "user", "content": prompt}
#                     ],
#                     max_tokens=max_tokens,
#                     temperature=1.0
#                 )
#             except self.openai.error.OpenAIError as e:
#                 time.sleep(retry_wait)
#         return {'choices': [{'message': {'content': ''}}]}

#     def _parse_message(self, msg):
#         try:
#             content = msg['choices'][0]['message']['content']
#         except (IndexError, KeyError):
#             content = ''
#         return content

#     def f(self, prompt, x):
#         msg = self._call_api(
#             prompt=prompt+x,
#             engine=self.engine,
#             max_tokens=self.max_tokens
#         )
#         evaluation = self._parse_message(msg)
#         return evaluation


class Checker(object):
    """A modified version of the Draft, Sketch, Prove proof-checking client.
    (https://github.com/albertqjiang/draft_sketch_prove/blob/main/autoformalization/checker.py)

    This checker supports Isabelle2022 via the new version of PISA
    (https://albertqjiang.github.io/Portal-to-ISAbelle/).

    It supports checking a miniF2F-style proof via `check`.

    Finally, it replaces `sledgehammer` with a call to `normalhammer`.
    """
    def __init__(self, working_dir, isa_path, theory_file, port=9000):
        sys.path.append(os.environ['PISA_PATH'])
        # try:
        from pisa_client import initialise_env
        self.initialise_env = initialise_env
        # except:
        #     print("Set $PISA_PATH to /yourpath/to/Portal-to-ISAbelle/src/main/python")

        self.working_dir = working_dir
        self.isa_path = isa_path
        self.theory_file = theory_file
        self.port = port

    def _initialize(self):
        env = self.initialise_env(
            self.port,
            isa_path=self.isa_path,
            theory_file_path=self.theory_file,
            working_directory=self.working_dir
        )
        return env

    def _exit(self, env):
        try:
            env.post('exit')
        except:
            print("env.post('exit') timed out")
            pass
        os.system("ps aux | grep Isabelle | awk '{print $2}' | xargs kill -9 > /dev/null 2>&1")
        os.system("ps aux | grep poly | awk '{print $2}' | xargs kill -9 > /dev/null 2>&1")

    def _parse_output(self, obs):
        """Parse the sledgehammer output, otherwise return an empty string"""
        if '<hammer>' in obs:
            output = obs.split('<hammer>')[0]
        else:
            output = ''
        return output

    def _run_step(self, step, i, tls_name, env):
        obs, reward, done, metadata = env.step_to_top_level_state(
            action=step,
            tls_name=tls_name,
            new_name='default_%d' % i
        )
        error = None
        if 'error:' in obs or 'Step error' in obs or 'Unknown error' in obs:
            error = obs

        print("Action {}: {}".format(i, step))
        print("State {}: {}".format(i, obs))
        print()
        return obs, reward, done, metadata, error

    def _run_sledgehammer(self, step, i, tls_name, env):
        # First try heuristics
        for heuristic in ['by auto', 'by simp', 'by blast', 'by fastforce', 'by force', 'by eval', 'by presburger', 'by sos', 'by arith', 'by linarith', 'by (auto simp: field_simps)']:
            step_ = step.replace('normalhammer', heuristic)
            obs, reward, done, metadata, error = self._run_step(step_, i, tls_name, env)
            if error is None:
                obs = '%s <hammer> %s' % (heuristic, obs)
                return obs, reward, done, metadata, error
        # Try sledgehammer
        out = self._run_step(step, i, tls_name, env)
        return out

    def check(self, statement_and_proof):
        # Initialize environment
        env = self._initialize()
        env.initialise()

        # Wrap and parse theorem
        theory = Checker.wrap_theorem(statement_and_proof)
        # theory = statement_and_proof
        steps = Checker.get_parsed(env, theory)

        result = self._check(env, steps)
        
        return result
    
    def get_parsed(env, theory, tls_name='default'):
        # HACK: the parsing doesn't work well with `normalhammer`, so we replace
        # all hammer calls with sorry, then replace sorry to normalhammer after parsing.
        theory = theory.replace('sledgehammer', 'sorry')
        theory = theory.replace('normalhammer', 'sorry')

        steps = env.post(f"<parse text> ${theory}")
        steps = steps.split('<SEP>')
        steps = [s for s in steps if s.strip() != '']
        # remove weird '$' step and whitespace steps
        steps = [s for s in steps if s != '$' and s.strip() != '']
        steps = [s.replace('sorry', 'normalhammer') for s in steps]
        return steps

    def _check(self, env, steps):
        done = False
        reason = ''
        success = False
        step_results = []
        tls_name = 'default'
        return_logs = []
        for i, step in enumerate(steps):
            try:
                time0 = time.time()
                if 'normalhammer' in step:
                    obs, reward, done, metadata, error = self._run_sledgehammer(step, i, tls_name, env)
                else:
                    obs, reward, done, metadata, error = self._run_step(step, i, tls_name, env)
                step_time = time.time() - time0
                step_results.append(dict(index=i, step=step, output=self._parse_output(obs), step_time=step_time))
                if error is not None:
                    reason = error
                    success = False
                    done = False
                    break
            except:
                # Timeout - end the proof attempt
                success = False
                done = False
                reason = 'timeout (%d)' % len(step_results)
                step_results.append(dict(index=i, step=step, output=''))
                break

            # Change when successful
            tls_name = 'default_%d' % i

            if tls_name == 'default_7':
                returned_string = env.get_fact_defintion(
                    name_of_tls=tls_name,
                    fact_name="test.min'_def"
                )
                print(returned_string)

            return_logs.append("Action {}: {}\n".format(i, step))
            return_logs.append("State {}: {}\n".format(i, obs))
            return_logs.append("\n")
        
        with open("facts_log.txt",'w') as f:
            for log in return_logs:
                f.write(log)

        if done and reward == 1.0:
            success = True

        result = {
            'success': success,
            'reason': reason,
            'num_steps': len(steps),
            'last_step': len(step_results),
            'step_results': step_results,
            'theorem_and_proof': self.reconstruct(step_results) if success else ''
        }
        # Exit environment
        self._exit(env)
        return result

    def get_c_func_fact(self, c_file_name):
        func_names = func_name_extract(c_file_name)
        c_file_base_name=os.path.basename(c_file_name)
        c_file_base_name = os.path.splitext(c_file_base_name)[0]

        # Initialize environment
        env = self._initialize()
        env.initialise()

        theory = """theory Interactive imports "AutoCorres.AutoCorres" \n begin
        external_file "{}"
        install_C_file "{}"
        autocorres "{}"

        context {} begin\n""".format(c_file_name,c_file_name,c_file_name,c_file_base_name)

        theory += """lemma "x \<ge> y" \n unfolding""" 
        for fun_name in func_names:
            theory += " {}'_def".format(fun_name)

        theory += """\n end"""

        # theory = statement_and_proof
        steps = Checker.get_parsed(env, theory)

        step_results = []
        tls_name = 'default'
        for i, step in enumerate(steps):
            try:
                time0 = time.time()
                if 'normalhammer' in step:
                    obs, reward, done, metadata, error = self._run_sledgehammer(step, i, tls_name, env)
                else:
                    obs, reward, done, metadata, error = self._run_step(step, i, tls_name, env)
                step_time = time.time() - time0
                step_results.append(dict(index=i, step=step, output=self._parse_output(obs), step_time=step_time))
                if error is not None:
                    break
            except:
                # Timeout - end the proof attempt
                step_results.append(dict(index=i, step=step, output=''))
                break

            # Change when successful
            tls_name = 'default_%d' % i

            if "unfolding" in step:
                returned_strings = {}
                for fun_name in func_names:
                    thm_name_full = " {}.{}'_def".format(c_file_base_name, fun_name)
                    thm_name = " {}'_def".format(fun_name)
                    returned_strings[thm_name] = env.get_fact_defintion(
                        name_of_tls=tls_name,
                        fact_name=thm_name_full
                    )[len(thm_name)+2:]
                break
        
        self._exit(env)
        return returned_strings
    
    @staticmethod
    def reconstruct(step_results):
        steps = []
        for step_result in step_results[1:]:
            if step_result['output'] != '':
                steps.append(step_result['output'].strip())
            else:
                steps.append(step_result['step'].strip())
        theorem_and_proof = '\n'.join(steps)
        return theorem_and_proof

    @staticmethod
    def wrap_theorem(theorem):
        # return 'theory Interactive imports HOL.HOL Complex_Main "AutoCorres.AutoCorres" "HOL-Library.Code_Target_Numeral" "HOL-Library.Sum_of_Squares" "Symmetric_Polynomials.Vieta" "HOL-Computational_Algebra.Computational_Algebra" "HOL-Number_Theory.Number_Theory" \n begin\n%s' % theorem
        return 'theory Interactive imports "AutoCorres.AutoCorres" \n begin\n%s' % theorem

    @staticmethod
    def get_parsed(env, theory, tls_name='default'):
        # HACK: the parsing doesn't work well with `normalhammer`, so we replace
        # all hammer calls with sorry, then replace sorry to normalhammer after parsing.
        theory = theory.replace('sledgehammer', 'sorry')
        theory = theory.replace('normalhammer', 'sorry')

        steps = env.post(f"<parse text> ${theory}")
        steps = steps.split('<SEP>')
        steps = [s for s in steps if s.strip() != '']
        # remove weird '$' step and whitespace steps
        steps = [s for s in steps if s != '$' and s.strip() != '']
        steps = [s.replace('sorry', 'normalhammer') for s in steps]
        return steps
